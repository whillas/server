#ifndef TENSORFLOW_SERVING_MODEL_SERVERS_PREDICTION_SERVICE_IMPL_H_
#define TENSORFLOW_SERVING_MODEL_SERVERS_PREDICTION_SERVICE_IMPL_H_

#include <grpc++/grpc++.h>
#include <nvtx3/nvToolsExt.h>
#include "src/core/grpc_service.grpc.pb.h"
#include "src/servers/shared_memory_manager.h"
#include "src/servers/tracer.h"
#include "triton/core/tritonserver.h"


#include <google/protobuf/arena.h>
#include <grpc++/alarm.h>
#include <nvtx3/nvToolsExt.h>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <list>
#include <map>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>
#include "grpc++/grpc++.h"
#include "grpc++/security/server_credentials.h"
#include "grpc++/server.h"
#include "grpc++/server_builder.h"
#include "grpc++/server_context.h"
#include "grpc++/support/status.h"
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"
#include "src/servers/classification.h"
#include "src/servers/common.h"
#include "triton/core/tritonserver.h"

namespace nvidia { namespace inferenceserver {

//
// AllocPayload
//
// Simple structure that carries the userp payload needed for
// allocation.
//
template <typename ResponseType>
struct AllocPayload {
  struct ShmInfo {
    ShmInfo(
        void* base, size_t byte_size, TRITONSERVER_MemoryType memory_type,
        int64_t memory_type_id)
        : base_(base), byte_size_(byte_size), memory_type_(memory_type),
          memory_type_id_(memory_type_id)
    {
    }
    void* base_;
    size_t byte_size_;
    TRITONSERVER_MemoryType memory_type_;
    int64_t memory_type_id_;
  };

  using TensorShmMap = std::unordered_map<std::string, ShmInfo>;
  using ClassificationMap = std::unordered_map<std::string, uint32_t>;

  explicit AllocPayload() : response_(nullptr) {}
  ~AllocPayload()
  {
    // Don't delete 'response_'.. it is owned by the InferHandlerState
  }

  ResponseType* response_;
  uint32_t response_alloc_count_;
  TensorShmMap shm_map_;
  ClassificationMap classification_map_;

  // Used to extend the lifetime of the serialized data in case
  // non-raw contents were provided in the request. Serialized data's
  // actual lifetime is that of the request whereas AllocPayload's
  // lifetime is that of a response... but it is convenient to keep it
  // here.
  std::list<std::string> serialized_data_;
};


//
// InferHandlerState
//
class InferHandlerState {
 public:
  explicit InferHandlerState(TRITONSERVER_Server* tritonserver)
      : tritonserver_(tritonserver)
  {
  }

  // Needed in the response handle for classification outputs.
  TRITONSERVER_Server* tritonserver_;

  // The state
  nvtxEventAttributes_t eventAttrib;
  nvtxRangeId_t id;

#ifdef TRITON_ENABLE_TRACING
  TraceManager* trace_manager_;
  TRITONSERVER_InferenceTrace* trace_;
  uint64_t trace_id_;
#endif  // TRITON_ENABLE_TRACING

  bool complete_;
  inference::ModelInferResponse* response_;
  TRITONSERVER_Error* err_;
  std::condition_variable cv_;
  std::mutex mutex_;

  // For inference requests the allocator payload, unused for other
  // requests.
  AllocPayload<inference::ModelInferResponse> alloc_payload_;
};


class InferenceServiceImpl final
    : public inference::GRPCInferenceService::Service {
 public:
   InferenceServiceImpl(
      std::shared_ptr<TRITONSERVER_Server>& tritonserver,  std::shared_ptr<SharedMemoryManager>& shm_manager);

  ~InferenceServiceImpl();

  ::grpc::Status ServerMetadata(
      ::grpc::ServerContext* context,
      const inference::ServerMetadataRequest* request,
      inference::ServerMetadataResponse* response) override;

  ::grpc::Status ModelMetadata(
      ::grpc::ServerContext* context,
      const inference::ModelMetadataRequest* request,
      inference::ModelMetadataResponse* response) override;

  ::grpc::Status ModelConfig(
      ::grpc::ServerContext* context,
      const inference::ModelConfigRequest* request,
      inference::ModelConfigResponse* response) override;


  ::grpc::Status ModelStatistics(
      ::grpc::ServerContext* context,
      const inference::ModelStatisticsRequest* request,
      inference::ModelStatisticsResponse* response) override;

  ::grpc::Status ModelInfer(
      ::grpc::ServerContext* context,
      const inference::ModelInferRequest* request,
      inference::ModelInferResponse* response) override;

 private:
  static void InferResponseComplete(
    TRITONSERVER_InferenceResponse* iresponse, const uint32_t flags,
    void* userp);
  static TRITONSERVER_Error* InferResponseCompleteCommon(
    TRITONSERVER_Server* server, TRITONSERVER_InferenceResponse* iresponse,
    inference::ModelInferResponse& response,
    const AllocPayload<inference::ModelInferResponse>& alloc_payload);


  std::shared_ptr<TRITONSERVER_Server> tritonserver_;
  TRITONSERVER_ResponseAllocator* allocator_;
  std::shared_ptr<SharedMemoryManager> shm_manager_;
};

}}  // namespace nvidia::inferenceserver

#endif  // TENSORFLOW_SERVING_MODEL_SERVERS_PREDICTION_SERVICE_IMPL_H_
