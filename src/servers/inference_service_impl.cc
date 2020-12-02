

#include "src/servers/inference_service_impl.h"

namespace nvidia { namespace inferenceserver {

namespace {


//
// GrpcStatusUtil
//
class GrpcStatusUtil {
 public:
  static void Populate(grpc::Status* status, TRITONSERVER_Error* err);
  static grpc::StatusCode CodeToStatus(TRITONSERVER_Error_Code code);
};

void
GrpcStatusUtil::Populate(grpc::Status* status, TRITONSERVER_Error* err)
{
  if (err == nullptr) {
    *status = grpc::Status::OK;
  } else {
    *status = grpc::Status(
        GrpcStatusUtil::CodeToStatus(TRITONSERVER_ErrorCode(err)),
        TRITONSERVER_ErrorMessage(err));
  }
}

grpc::StatusCode
GrpcStatusUtil::CodeToStatus(TRITONSERVER_Error_Code code)
{
  // GRPC status codes:
  // https://github.com/grpc/grpc/blob/master/include/grpc/impl/codegen/status.h
  switch (code) {
    case TRITONSERVER_ERROR_UNKNOWN:
      return grpc::StatusCode::UNKNOWN;
    case TRITONSERVER_ERROR_INTERNAL:
      return grpc::StatusCode::INTERNAL;
    case TRITONSERVER_ERROR_NOT_FOUND:
      return grpc::StatusCode::NOT_FOUND;
    case TRITONSERVER_ERROR_INVALID_ARG:
      return grpc::StatusCode::INVALID_ARGUMENT;
    case TRITONSERVER_ERROR_UNAVAILABLE:
      return grpc::StatusCode::UNAVAILABLE;
    case TRITONSERVER_ERROR_UNSUPPORTED:
      return grpc::StatusCode::UNIMPLEMENTED;
    case TRITONSERVER_ERROR_ALREADY_EXISTS:
      return grpc::StatusCode::ALREADY_EXISTS;
  }

  return grpc::StatusCode::UNKNOWN;
}


template <typename ShmMapType>
TRITONSERVER_Error*
ResponseAllocatorHelper(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, inference::ModelInferResponse* response,
    const ShmMapType& shm_map, void** buffer, void** buffer_userp,
    TRITONSERVER_MemoryType* actual_memory_type, int64_t* actual_memory_type_id)
{
  *buffer = nullptr;
  *buffer_userp = nullptr;
  *actual_memory_type = preferred_memory_type;
  *actual_memory_type_id = preferred_memory_type_id;

  // We add an output contents even if the 'byte_size' == 0 because we
  // expect to have a contents for every output.
  inference::ModelInferResponse::InferOutputTensor* output_tensor =
      response->add_outputs();
  output_tensor->set_name(tensor_name);
  std::string* raw_output = response->add_raw_output_contents();

  if (byte_size > 0) {
    const auto& pr = shm_map.find(tensor_name);
    if (pr != shm_map.end()) {
      // The output is in shared memory so check that shared memory
      // size is at least large enough for the output.
      if (byte_size > pr->second.byte_size_) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            std::string(
                "shared memory size specified with the request for output '" +
                std::string(tensor_name) + "' (" +
                std::to_string(pr->second.byte_size_) +
                " bytes) should be at least " + std::to_string(byte_size) +
                " bytes to hold the results")
                .c_str());
      }

      *buffer = const_cast<void*>(pr->second.base_);
      *actual_memory_type = pr->second.memory_type_;
      *actual_memory_type_id = pr->second.memory_type_id_;

      LOG_VERBOSE(1) << "GRPC: using shared-memory for '" << tensor_name
                     << "', size: " << byte_size << ", addr: " << *buffer;
      return nullptr;  // Success
    }

    // Not using shared memory so allocate a buffer. The buffer we
    // create is directly in the response protobuf so we can't
    // allocate any type other than CPU.
    //
    // FIXME we could use pinned CPU memory here.
    if (*actual_memory_type != TRITONSERVER_MEMORY_CPU) {
      LOG_VERBOSE(1) << "GRPC: unable to provide '" << tensor_name << "' in "
                     << TRITONSERVER_MemoryTypeString(*actual_memory_type)
                     << ", will use "
                     << TRITONSERVER_MemoryTypeString(TRITONSERVER_MEMORY_CPU);
      *actual_memory_type = TRITONSERVER_MEMORY_CPU;
      *actual_memory_type_id = 0;
    }

    raw_output->resize(byte_size);
    *buffer = static_cast<void*>(&((*raw_output)[0]));

    LOG_VERBOSE(1) << "GRPC: using buffer for '" << tensor_name
                   << "', size: " << byte_size << ", addr: " << *buffer;
  }

  return nullptr;  // Success
}

TRITONSERVER_Error*
InferResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  AllocPayload<inference::ModelInferResponse>* payload =
      reinterpret_cast<AllocPayload<inference::ModelInferResponse>*>(userp);

  // ModelInfer RPC expects exactly one response per request. Hence,
  // will be creating and using just one response object.
  inference::ModelInferResponse* response = payload->response_;
  return ResponseAllocatorHelper<
      AllocPayload<inference::ModelInferResponse>::TensorShmMap>(
      allocator, tensor_name, byte_size, preferred_memory_type,
      preferred_memory_type_id, response, payload->shm_map_, buffer,
      buffer_userp, actual_memory_type, actual_memory_type_id);
}

TRITONSERVER_Error*
InferResponseFree(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  LOG_VERBOSE(1) << "GRPC free: "
                 << "size " << byte_size << ", addr " << buffer;

  // Don't do anything when releasing a buffer since InferResponseAlloc
  // wrote directly into the response protobuf.
  return nullptr;  // Success
}

template <typename TensorType>
TRITONSERVER_Error*
ParseSharedMemoryParams(
    const TensorType& tensor, bool* has_shared_memory, std::string* region_name,
    int64_t* offset, size_t* byte_size)
{
  *has_shared_memory = false;
  *offset = 0 /* default value */;
  const auto& region_it = tensor.parameters().find("shared_memory_region");
  if (region_it != tensor.parameters().end()) {
    *has_shared_memory = true;
    const auto& infer_param = region_it->second;
    if (infer_param.parameter_choice_case() !=
        inference::InferParameter::ParameterChoiceCase::kStringParam) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "invalid value type for 'shared_memory_region' parameter for "
              "tensor '" +
              tensor.name() + "', expected string_param.")
              .c_str());
    }
    *region_name = infer_param.string_param();
  }

  const auto& offset_it = tensor.parameters().find("shared_memory_offset");
  if (offset_it != tensor.parameters().end()) {
    if (!*has_shared_memory) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "'shared_memory_offset' can not be specified without "
              "'shared_memory_region' parameter for tensor '" +
              tensor.name() + "'")
              .c_str());
    }
    const auto& infer_param = offset_it->second;
    if (infer_param.parameter_choice_case() !=
        inference::InferParameter::ParameterChoiceCase::kInt64Param) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "invalid value type for 'shared_memory_offset' parameter for "
              "tensor '" +
              tensor.name() + "', expected int64_param.")
              .c_str());
    }
    *offset = infer_param.int64_param();
  }

  const auto& bs_it = tensor.parameters().find("shared_memory_byte_size");
  if (bs_it != tensor.parameters().end()) {
    if (!*has_shared_memory) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "'shared_memory_byte_size' can not be specified without "
              "'shared_memory_region' parameter for tensor '" +
              tensor.name() + "'")
              .c_str());
    }
    const auto& infer_param = bs_it->second;
    if (infer_param.parameter_choice_case() !=
        inference::InferParameter::ParameterChoiceCase::kInt64Param) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "invalid value type for 'shared_memory_byte_size' parameter "
              "for "
              "tensor '" +
              tensor.name() + "', expected int64_param.")
              .c_str());
    }
    *byte_size = infer_param.int64_param();
  } else {
    if (*has_shared_memory) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "'shared_memory_byte_size' must be specified along with "
              "'shared_memory_region' parameter for tensor '" +
              tensor.name() + "'")
              .c_str());
    }
  }

  return nullptr;
}


TRITONSERVER_Error*
ParseClassificationParams(
    const inference::ModelInferRequest::InferRequestedOutputTensor& output,
    bool* has_classification, uint32_t* classification_count)
{
  *has_classification = false;

  const auto& class_it = output.parameters().find("classification");
  if (class_it != output.parameters().end()) {
    *has_classification = true;

    const auto& param = class_it->second;
    if (param.parameter_choice_case() !=
        inference::InferParameter::ParameterChoiceCase::kInt64Param) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "invalid value type for 'classification' parameter, expected "
          "int64_param");
    }

    const int64_t cnt = param.int64_param();
    if (cnt <= 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "invalid value for 'classification' parameter, expected >= 0");
    }

    *classification_count = cnt;
  }

  return nullptr;  // success
}

template <typename ResponseType>
TRITONSERVER_Error*
InferAllocatorPayload(
    const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
    const std::shared_ptr<SharedMemoryManager>& shm_manager,
    const inference::ModelInferRequest& request,
    std::list<std::string>&& serialized_data, ResponseType* response,
    AllocPayload<ResponseType>* alloc_payload)
{
  alloc_payload->response_ = response;
  alloc_payload->shm_map_.clear();
  alloc_payload->classification_map_.clear();
  alloc_payload->serialized_data_ = std::move(serialized_data);

  // If any of the outputs use shared memory, then we must calculate
  // the memory address for that output and store it in the allocator
  // payload so that it is available when the allocation callback is
  // invoked.
  for (const auto& io : request.outputs()) {
    std::string region_name;
    int64_t offset;
    size_t byte_size;
    bool has_shared_memory;
    RETURN_IF_ERR(ParseSharedMemoryParams<
                  inference::ModelInferRequest::InferRequestedOutputTensor>(
        io, &has_shared_memory, &region_name, &offset, &byte_size));

    bool has_classification;
    uint32_t classification_count;
    RETURN_IF_ERR(ParseClassificationParams(
        io, &has_classification, &classification_count));

    if (has_shared_memory && has_classification) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "output can't set both 'shared_memory_region' and "
          "'classification'");
    }

    if (has_shared_memory) {
      void* base;
      TRITONSERVER_MemoryType memory_type;
      int64_t memory_type_id;
      RETURN_IF_ERR(shm_manager->GetMemoryInfo(
          region_name, offset, &base, &memory_type, &memory_type_id));

      alloc_payload->shm_map_.emplace(
          io.name(), typename AllocPayload<ResponseType>::ShmInfo(
                         base, byte_size, memory_type, memory_type_id));
    } else if (has_classification) {
      alloc_payload->classification_map_.emplace(
          io.name(), classification_count);
    }
  }

  return nullptr;  // Success
}

TRITONSERVER_Error*
InferGRPCToInputHelper(
    const std::string& input_name, const std::string& model_name,
    const TRITONSERVER_DataType tensor_dt, const TRITONSERVER_DataType input_dt,
    const size_t binary_data_byte_size)
{
  if (binary_data_byte_size != 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string(
            "unexpected explicit tensor data for input tensor '" + input_name +
            "' for model '" + model_name +
            "', binary data was already supplied.")
            .c_str());
  }

  if (tensor_dt != input_dt) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string(
            "unexpected explicit tensor data for input tensor '" + input_name +
            "' for model '" + model_name + "' of type '" +
            TRITONSERVER_DataTypeString(tensor_dt) + "', expected datatype '" +
            TRITONSERVER_DataTypeString(input_dt) + "'")
            .c_str());
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
InferGRPCToInput(
    const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
    const std::shared_ptr<SharedMemoryManager>& shm_manager,
    const inference::ModelInferRequest& request,
    std::list<std::string>* serialized_data,
    TRITONSERVER_InferenceRequest* inference_request)
{
  // Verify that the batch-byte-size of each input matches the size of
  // the provided tensor data (provided raw or from shared memory)
  int index = 0;
  for (const auto& io : request.inputs()) {
    const void* base;
    size_t byte_size = 0;
    TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t memory_type_id = 0;

    std::string region_name;
    int64_t offset;
    bool has_shared_memory;
    RETURN_IF_ERR(
        ParseSharedMemoryParams<inference::ModelInferRequest::InferInputTensor>(
            io, &has_shared_memory, &region_name, &offset, &byte_size));

    if (has_shared_memory) {
      if (io.has_contents()) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            std::string(
                "unexpected 'content' provided when using shared memory "
                "for "
                "input tensor '" +
                io.name() + "' for model '" + request.model_name() + "'")
                .c_str());
      }
      void* tmp;
      RETURN_IF_ERR(shm_manager->GetMemoryInfo(
          region_name, offset, &tmp, &memory_type, &memory_type_id));
      base = tmp;
    } else {
      if (io.has_contents() && (!request.raw_input_contents().empty())) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            std::string(
                "contents field must not be specified when using "
                "raw_input_contents for '" +
                io.name() + "' for model '" + request.model_name() + "'")
                .c_str());
      } else if (io.has_contents()) {
        // Check the presence of explicit tensors
        TRITONSERVER_DataType dtype =
            TRITONSERVER_StringToDataType(io.datatype().c_str());
        const size_t elem_byte_size = TRITONSERVER_DataTypeByteSize(dtype);
        if (io.contents().bool_contents_size() != 0) {
          RETURN_IF_ERR(InferGRPCToInputHelper(
              io.name(), request.model_name(), TRITONSERVER_TYPE_BOOL, dtype,
              byte_size));
          base = (const void*)io.contents().bool_contents().data();
          byte_size = io.contents().bool_contents_size() * elem_byte_size;
        }

        if (io.contents().int_contents_size() != 0) {
          if (dtype == TRITONSERVER_TYPE_INT8) {
            RETURN_IF_ERR(InferGRPCToInputHelper(
                io.name(), request.model_name(), TRITONSERVER_TYPE_INT8, dtype,
                byte_size));
            serialized_data->emplace_back();
            auto& serialized = serialized_data->back();
            serialized.reserve(
                io.contents().int_contents_size() * elem_byte_size);
            for (const auto& element : io.contents().int_contents()) {
              // Assuming the system is little-endian, picking the
              // least significant byte of 32-bit integer as a
              // int8 element
              serialized.append(
                  reinterpret_cast<const char*>(&element), elem_byte_size);
            }
            base = serialized.c_str();
            byte_size = serialized.size();
          } else if (dtype == TRITONSERVER_TYPE_INT16) {
            RETURN_IF_ERR(InferGRPCToInputHelper(
                io.name(), request.model_name(), TRITONSERVER_TYPE_INT16, dtype,
                byte_size));
            serialized_data->emplace_back();
            auto& serialized = serialized_data->back();
            serialized.reserve(
                io.contents().int_contents_size() * elem_byte_size);
            for (const auto& element : io.contents().int_contents()) {
              // Assuming the system is little-endian, picking the
              // least 2 significant bytes of 32-bit integer as a
              // int16 element
              serialized.append(
                  reinterpret_cast<const char*>(&element), elem_byte_size);
            }
            base = serialized.c_str();
            byte_size = serialized.size();
          } else {
            RETURN_IF_ERR(InferGRPCToInputHelper(
                io.name(), request.model_name(), TRITONSERVER_TYPE_INT32, dtype,
                byte_size));
            base = (const void*)io.contents().int_contents().data();
            byte_size = io.contents().int_contents_size() * elem_byte_size;
          }
        }

        if (io.contents().int64_contents_size() != 0) {
          RETURN_IF_ERR(InferGRPCToInputHelper(
              io.name(), request.model_name(), TRITONSERVER_TYPE_INT64, dtype,
              byte_size));
          base = (const void*)io.contents().int64_contents().data();
          byte_size = io.contents().int64_contents_size() * elem_byte_size;
        }

        if (io.contents().uint_contents_size() != 0) {
          if (dtype == TRITONSERVER_TYPE_UINT8) {
            RETURN_IF_ERR(InferGRPCToInputHelper(
                io.name(), request.model_name(), TRITONSERVER_TYPE_UINT8, dtype,
                byte_size));
            serialized_data->emplace_back();
            auto& serialized = serialized_data->back();
            serialized.reserve(
                io.contents().uint_contents_size() * elem_byte_size);
            for (const auto& element : io.contents().uint_contents()) {
              // Assuming the system is little-endian, picking the
              // least significant byte of 32-bit unsigned integer as a
              // uint8 element
              serialized.append(
                  reinterpret_cast<const char*>(&element), elem_byte_size);
            }
            base = serialized.c_str();
            byte_size = serialized.size();
          } else if (dtype == TRITONSERVER_TYPE_UINT16) {
            RETURN_IF_ERR(InferGRPCToInputHelper(
                io.name(), request.model_name(), TRITONSERVER_TYPE_UINT16,
                dtype, byte_size));
            serialized_data->emplace_back();
            auto& serialized = serialized_data->back();
            serialized.reserve(
                io.contents().uint_contents_size() * elem_byte_size);
            for (const auto& element : io.contents().uint_contents()) {
              // Assuming the system is little-endian, picking the
              // least 2 significant bytes of 32-bit integer as a
              // uint16 element
              serialized.append(
                  reinterpret_cast<const char*>(&element), elem_byte_size);
            }
            base = serialized.c_str();
            byte_size = serialized.size();
          } else {
            RETURN_IF_ERR(InferGRPCToInputHelper(
                io.name(), request.model_name(), TRITONSERVER_TYPE_UINT32,
                dtype, byte_size));
            base = (const void*)io.contents().int_contents().data();
            byte_size = io.contents().int_contents_size() * elem_byte_size;
          }
        }

        if (io.contents().uint64_contents_size() != 0) {
          RETURN_IF_ERR(InferGRPCToInputHelper(
              io.name(), request.model_name(), TRITONSERVER_TYPE_UINT64, dtype,
              byte_size));
          base = (const void*)io.contents().uint64_contents().data();
          byte_size = io.contents().uint64_contents_size() * elem_byte_size;
        }

        if (io.contents().fp32_contents_size() != 0) {
          RETURN_IF_ERR(InferGRPCToInputHelper(
              io.name(), request.model_name(), TRITONSERVER_TYPE_FP32, dtype,
              byte_size));
          base = (const void*)io.contents().fp32_contents().data();
          byte_size = io.contents().fp32_contents_size() * elem_byte_size;
        }

        if (io.contents().fp64_contents_size() != 0) {
          RETURN_IF_ERR(InferGRPCToInputHelper(
              io.name(), request.model_name(), TRITONSERVER_TYPE_FP64, dtype,
              byte_size));
          base = (const void*)io.contents().fp64_contents().data();
          byte_size = io.contents().fp64_contents_size() * elem_byte_size;
        }

        if (io.contents().byte_contents_size() != 0) {
          RETURN_IF_ERR(InferGRPCToInputHelper(
              io.name(), request.model_name(), TRITONSERVER_TYPE_BYTES, dtype,
              byte_size));

          serialized_data->emplace_back();
          auto& serialized = serialized_data->back();

          // Serialize the output tensor strings. Each string is
          // serialized as a 4-byte length followed by the string itself
          // with no null-terminator.
          for (const auto& element : io.contents().byte_contents()) {
            uint32_t len{(uint32_t)element.size()};
            serialized.append(
                reinterpret_cast<const char*>(&len), sizeof(uint32_t));
            if (element.size() > 0) {
              serialized.append(element.c_str(), len);
            }
          }
          base = serialized.c_str();
          byte_size = serialized.size();
        }
      } else if (request.raw_input_contents().size() > index) {
        // Try to read the raw contents if available
        const std::string& raw = request.raw_input_contents()[index++];
        base = raw.c_str();
        byte_size = raw.size();
      } else {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            std::string(
                "unable to find data for input tensor '" + io.name() +
                "' for model '" + request.model_name() + "' in request.")
                .c_str());
      }
    }

    RETURN_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
        inference_request, io.name().c_str(), base, byte_size, memory_type,
        memory_type_id));
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
SetInferenceRequestMetadata(
    TRITONSERVER_InferenceRequest* inference_request,
    const inference::ModelInferRequest& request)
{
  RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetId(
      inference_request, request.id().c_str()));

  uint32_t flags = 0;
  for (auto param : request.parameters()) {
    if (param.first.compare("sequence_id") == 0) {
      const auto& infer_param = param.second;
      if (infer_param.parameter_choice_case() !=
          inference::InferParameter::ParameterChoiceCase::kInt64Param) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "invalid value type for 'sequence_id' parameter, expected "
            "int64_param.");
      }
      RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetCorrelationId(
          inference_request, infer_param.int64_param()));
    } else if (param.first.compare("sequence_start") == 0) {
      const auto& infer_param = param.second;
      if (infer_param.parameter_choice_case() !=
          inference::InferParameter::ParameterChoiceCase::kBoolParam) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "invalid value type for 'sequence_start' parameter, expected "
            "bool_param.");
      }
      if (infer_param.bool_param()) {
        flags |= TRITONSERVER_REQUEST_FLAG_SEQUENCE_START;
      }
    } else if (param.first.compare("sequence_end") == 0) {
      const auto& infer_param = param.second;
      if (infer_param.parameter_choice_case() !=
          inference::InferParameter::ParameterChoiceCase::kBoolParam) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "invalid value type for 'sequence_end' parameter, expected "
            "bool_param.");
      }
      if (infer_param.bool_param()) {
        flags |= TRITONSERVER_REQUEST_FLAG_SEQUENCE_END;
      }
    } else if (param.first.compare("priority") == 0) {
      const auto& infer_param = param.second;
      if (infer_param.parameter_choice_case() !=
          inference::InferParameter::ParameterChoiceCase::kInt64Param) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "invalid value type for 'priority' parameter, expected "
            "int64_param.");
      }
      RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetPriority(
          inference_request, infer_param.int64_param()));

    } else if (param.first.compare("timeout") == 0) {
      const auto& infer_param = param.second;
      if (infer_param.parameter_choice_case() !=
          inference::InferParameter::ParameterChoiceCase::kInt64Param) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "invalid value type for 'timeout' parameter, expected "
            "int64_param.");
      }
      RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetTimeoutMicroseconds(
          inference_request, infer_param.int64_param()));
    }
  }

  RETURN_IF_ERR(
      TRITONSERVER_InferenceRequestSetFlags(inference_request, flags));

  for (const auto& input : request.inputs()) {
    RETURN_IF_ERR(TRITONSERVER_InferenceRequestAddInput(
        inference_request, input.name().c_str(),
        TRITONSERVER_StringToDataType(input.datatype().c_str()),
        input.shape().data(), input.shape_size()));
  }

  for (const auto& output : request.outputs()) {
    RETURN_IF_ERR(TRITONSERVER_InferenceRequestAddRequestedOutput(
        inference_request, output.name().c_str()));
  }

  return nullptr;  // Success
}

void
InferRequestComplete(
    TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
  LOG_VERBOSE(1) << "ModelInferHandler::InferRequestComplete";

  if ((flags & TRITONSERVER_REQUEST_RELEASE_ALL) != 0) {
    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceRequestDelete(request),
        "deleting GRPC inference request");
  }
}
}  // namespace


InferenceServiceImpl::InferenceServiceImpl(
      std::shared_ptr<TRITONSERVER_Server>& tritonserver,  std::shared_ptr<SharedMemoryManager>& shm_manager)
      : tritonserver_(tritonserver), shm_manager_(shm_manager)
  {
      // Create the allocator that will be used to allocate buffers for
    // the result tensors.
    FAIL_IF_ERR(
        TRITONSERVER_ResponseAllocatorNew(
            &allocator_, InferResponseAlloc, InferResponseFree,
            nullptr),
        "creating inference response allocator");
  }

InferenceServiceImpl::~InferenceServiceImpl() {
   LOG_TRITONSERVER_ERROR(
        TRITONSERVER_ResponseAllocatorDelete(allocator_),
        "deleting response allocator");
}

::grpc::Status
InferenceServiceImpl::ServerMetadata(
    ::grpc::ServerContext* context,
    const inference::ServerMetadataRequest* request,
    inference::ServerMetadataResponse* response)
{
  grpc::Status status;
  TRITONSERVER_Message* server_metadata_message = nullptr;
  TRITONSERVER_Error* err = TRITONSERVER_ServerMetadata(
      tritonserver_.get(), &server_metadata_message);
  GOTO_IF_ERR(err, earlyexit);

  const char* buffer;
  size_t byte_size;
  err = TRITONSERVER_MessageSerializeToJson(
      server_metadata_message, &buffer, &byte_size);
  GOTO_IF_ERR(err, earlyexit);

  {
    triton::common::TritonJson::Value server_metadata_json;
    err = server_metadata_json.Parse(buffer, byte_size);
    GOTO_IF_ERR(err, earlyexit);

    const char* name;
    size_t namelen;
    err = server_metadata_json.MemberAsString("name", &name, &namelen);
    GOTO_IF_ERR(err, earlyexit);

    const char* version;
    size_t versionlen;
    err = server_metadata_json.MemberAsString("version", &version, &versionlen);
    GOTO_IF_ERR(err, earlyexit);

    response->set_name(std::string(name, namelen));
    response->set_version(std::string(version, versionlen));

    if (server_metadata_json.Find("extensions")) {
      triton::common::TritonJson::Value extensions_json;
      err = server_metadata_json.MemberAsArray("extensions", &extensions_json);
      GOTO_IF_ERR(err, earlyexit);

      for (size_t idx = 0; idx < extensions_json.ArraySize(); ++idx) {
        const char* ext;
        size_t extlen;
        err = extensions_json.IndexAsString(idx, &ext, &extlen);
        GOTO_IF_ERR(err, earlyexit);
        response->add_extensions(std::string(ext, extlen));
      }
    }
    TRITONSERVER_MessageDelete(server_metadata_message);
  }

earlyexit:
  GrpcStatusUtil::Populate(&status, err);
  TRITONSERVER_ErrorDelete(err);

  return status;
}

::grpc::Status
InferenceServiceImpl::ModelMetadata(
    ::grpc::ServerContext* context,
    const inference::ModelMetadataRequest* request,
    inference::ModelMetadataResponse* response)
{
  grpc::Status status;

  int64_t requested_model_version;
  auto err =
      GetModelVersionFromString(request->version(), &requested_model_version);
  GOTO_IF_ERR(err, earlyexit);

  {
    TRITONSERVER_Message* model_metadata_message = nullptr;
    err = TRITONSERVER_ServerModelMetadata(
        tritonserver_.get(), request->name().c_str(), requested_model_version,
        &model_metadata_message);
    GOTO_IF_ERR(err, earlyexit);

    const char* buffer;
    size_t byte_size;
    err = TRITONSERVER_MessageSerializeToJson(
        model_metadata_message, &buffer, &byte_size);
    GOTO_IF_ERR(err, earlyexit);

    triton::common::TritonJson::Value model_metadata_json;
    err = model_metadata_json.Parse(buffer, byte_size);
    GOTO_IF_ERR(err, earlyexit);

    const char* name;
    size_t namelen;
    err = model_metadata_json.MemberAsString("name", &name, &namelen);
    GOTO_IF_ERR(err, earlyexit);

    response->set_name(std::string(name, namelen));

    if (model_metadata_json.Find("versions")) {
      triton::common::TritonJson::Value versions_json;
      err = model_metadata_json.MemberAsArray("versions", &versions_json);
      GOTO_IF_ERR(err, earlyexit);

      for (size_t idx = 0; idx < versions_json.ArraySize(); ++idx) {
        const char* version;
        size_t versionlen;
        err = versions_json.IndexAsString(idx, &version, &versionlen);
        GOTO_IF_ERR(err, earlyexit);
        response->add_versions(std::string(version, versionlen));
      }
    }

    const char* platform;
    size_t platformlen;
    err =
        model_metadata_json.MemberAsString("platform", &platform, &platformlen);
    GOTO_IF_ERR(err, earlyexit);
    response->set_platform(std::string(platform, platformlen));

    if (model_metadata_json.Find("inputs")) {
      triton::common::TritonJson::Value inputs_json;
      err = model_metadata_json.MemberAsArray("inputs", &inputs_json);
      GOTO_IF_ERR(err, earlyexit);

      for (size_t idx = 0; idx < inputs_json.ArraySize(); ++idx) {
        triton::common::TritonJson::Value io_json;
        err = inputs_json.IndexAsObject(idx, &io_json);
        GOTO_IF_ERR(err, earlyexit);

        inference::ModelMetadataResponse::TensorMetadata* io =
            response->add_inputs();

        const char* name;
        size_t namelen;
        err = io_json.MemberAsString("name", &name, &namelen);
        GOTO_IF_ERR(err, earlyexit);

        const char* datatype;
        size_t datatypelen;
        err = io_json.MemberAsString("datatype", &datatype, &datatypelen);
        GOTO_IF_ERR(err, earlyexit);

        io->set_name(std::string(name, namelen));
        io->set_datatype(std::string(datatype, datatypelen));

        if (io_json.Find("shape")) {
          triton::common::TritonJson::Value shape_json;
          err = io_json.MemberAsArray("shape", &shape_json);
          GOTO_IF_ERR(err, earlyexit);

          for (size_t sidx = 0; sidx < shape_json.ArraySize(); ++sidx) {
            int64_t d;
            err = shape_json.IndexAsInt(sidx, &d);
            GOTO_IF_ERR(err, earlyexit);

            io->add_shape(d);
          }
        }
      }
    }

    if (model_metadata_json.Find("outputs")) {
      triton::common::TritonJson::Value outputs_json;
      err = model_metadata_json.MemberAsArray("outputs", &outputs_json);
      GOTO_IF_ERR(err, earlyexit);

      for (size_t idx = 0; idx < outputs_json.ArraySize(); ++idx) {
        triton::common::TritonJson::Value io_json;
        err = outputs_json.IndexAsObject(idx, &io_json);
        GOTO_IF_ERR(err, earlyexit);

        inference::ModelMetadataResponse::TensorMetadata* io =
            response->add_outputs();

        const char* name;
        size_t namelen;
        err = io_json.MemberAsString("name", &name, &namelen);
        GOTO_IF_ERR(err, earlyexit);

        const char* datatype;
        size_t datatypelen;
        err = io_json.MemberAsString("datatype", &datatype, &datatypelen);
        GOTO_IF_ERR(err, earlyexit);

        io->set_name(std::string(name, namelen));
        io->set_datatype(std::string(datatype, datatypelen));

        if (io_json.Find("shape")) {
          triton::common::TritonJson::Value shape_json;
          err = io_json.MemberAsArray("shape", &shape_json);
          GOTO_IF_ERR(err, earlyexit);

          for (size_t sidx = 0; sidx < shape_json.ArraySize(); ++sidx) {
            int64_t d;
            err = shape_json.IndexAsInt(sidx, &d);
            GOTO_IF_ERR(err, earlyexit);

            io->add_shape(d);
          }
        }
      }
    }

    TRITONSERVER_MessageDelete(model_metadata_message);
  }

earlyexit:
  GrpcStatusUtil::Populate(&status, err);
  TRITONSERVER_ErrorDelete(err);

  return status;
}

::grpc::Status
InferenceServiceImpl::ModelConfig(
    ::grpc::ServerContext* context,
    const inference::ModelConfigRequest* request,
    inference::ModelConfigResponse* response)
{
  grpc::Status status;

  int64_t requested_model_version;
  auto err =
      GetModelVersionFromString(request->version(), &requested_model_version);
  if (err == nullptr) {
    TRITONSERVER_Message* model_config_message = nullptr;
    err = TRITONSERVER_ServerModelConfig(
        tritonserver_.get(), request->name().c_str(), requested_model_version,
        1 /* config_version */, &model_config_message);
    if (err == nullptr) {
      const char* buffer;
      size_t byte_size;
      err = TRITONSERVER_MessageSerializeToJson(
          model_config_message, &buffer, &byte_size);
      if (err == nullptr) {
        ::google::protobuf::util::JsonStringToMessage(
            {buffer, (int)byte_size}, response->mutable_config());
      }
      TRITONSERVER_MessageDelete(model_config_message);
    }
  }

  GrpcStatusUtil::Populate(&status, err);
  TRITONSERVER_ErrorDelete(err);

  return status;
}


::grpc::Status
InferenceServiceImpl::ModelStatistics(
    ::grpc::ServerContext* context,
    const inference::ModelStatisticsRequest* request,
    inference::ModelStatisticsResponse* response)
{
  grpc::Status status;

#ifdef TRITON_ENABLE_STATS
  triton::common::TritonJson::Value model_stats_json;

  int64_t requested_model_version;
  auto err =
      GetModelVersionFromString(request->version(), &requested_model_version);
  GOTO_IF_ERR(err, earlyexit);

  {
    TRITONSERVER_Message* model_stats_message = nullptr;
    err = TRITONSERVER_ServerModelStatistics(
        tritonserver_.get(), request->name().c_str(), requested_model_version,
        &model_stats_message);
    GOTO_IF_ERR(err, earlyexit);

    const char* buffer;
    size_t byte_size;
    err = TRITONSERVER_MessageSerializeToJson(
        model_stats_message, &buffer, &byte_size);
    GOTO_IF_ERR(err, earlyexit);

    err = model_stats_json.Parse(buffer, byte_size);
    GOTO_IF_ERR(err, earlyexit);

    TRITONSERVER_MessageDelete(model_stats_message);
  }

  if (model_stats_json.Find("model_stats")) {
    triton::common::TritonJson::Value stats_json;
    err = model_stats_json.MemberAsArray("model_stats", &stats_json);
    GOTO_IF_ERR(err, earlyexit);

    for (size_t idx = 0; idx < stats_json.ArraySize(); ++idx) {
      triton::common::TritonJson::Value model_stat;
      err = stats_json.IndexAsObject(idx, &model_stat);
      GOTO_IF_ERR(err, earlyexit);

      auto statistics = response->add_model_stats();

      const char* name;
      size_t namelen;
      err = model_stat.MemberAsString("name", &name, &namelen);
      GOTO_IF_ERR(err, earlyexit);

      const char* version;
      size_t versionlen;
      err = model_stat.MemberAsString("version", &version, &versionlen);
      GOTO_IF_ERR(err, earlyexit);

      statistics->set_name(std::string(name, namelen));
      statistics->set_version(std::string(version, versionlen));

      uint64_t ucnt;
      err = model_stat.MemberAsUInt("last_inference", &ucnt);
      GOTO_IF_ERR(err, earlyexit);
      statistics->set_last_inference(ucnt);

      err = model_stat.MemberAsUInt("inference_count", &ucnt);
      GOTO_IF_ERR(err, earlyexit);
      statistics->set_inference_count(ucnt);

      err = model_stat.MemberAsUInt("execution_count", &ucnt);
      GOTO_IF_ERR(err, earlyexit);
      statistics->set_execution_count(ucnt);

      triton::common::TritonJson::Value infer_stats_json;
      err = model_stat.MemberAsObject("inference_stats", &infer_stats_json);
      GOTO_IF_ERR(err, earlyexit);

      {
        triton::common::TritonJson::Value success_json;
        err = infer_stats_json.MemberAsObject("success", &success_json);
        GOTO_IF_ERR(err, earlyexit);

        err = success_json.MemberAsUInt("count", &ucnt);
        GOTO_IF_ERR(err, earlyexit);
        statistics->mutable_inference_stats()->mutable_success()->set_count(
            ucnt);
        err = success_json.MemberAsUInt("ns", &ucnt);
        GOTO_IF_ERR(err, earlyexit);
        statistics->mutable_inference_stats()->mutable_success()->set_ns(ucnt);
      }

      {
        triton::common::TritonJson::Value fail_json;
        err = infer_stats_json.MemberAsObject("fail", &fail_json);
        GOTO_IF_ERR(err, earlyexit);

        err = fail_json.MemberAsUInt("count", &ucnt);
        GOTO_IF_ERR(err, earlyexit);
        statistics->mutable_inference_stats()->mutable_fail()->set_count(ucnt);
        err = fail_json.MemberAsUInt("ns", &ucnt);
        GOTO_IF_ERR(err, earlyexit);
        statistics->mutable_inference_stats()->mutable_fail()->set_ns(ucnt);
      }

      {
        triton::common::TritonJson::Value queue_json;
        err = infer_stats_json.MemberAsObject("queue", &queue_json);
        GOTO_IF_ERR(err, earlyexit);

        err = queue_json.MemberAsUInt("count", &ucnt);
        GOTO_IF_ERR(err, earlyexit);
        statistics->mutable_inference_stats()->mutable_queue()->set_count(ucnt);
        err = queue_json.MemberAsUInt("ns", &ucnt);
        GOTO_IF_ERR(err, earlyexit);
        statistics->mutable_inference_stats()->mutable_queue()->set_ns(ucnt);
      }

      {
        triton::common::TritonJson::Value compute_input_json;
        err = infer_stats_json.MemberAsObject(
            "compute_input", &compute_input_json);
        GOTO_IF_ERR(err, earlyexit);

        err = compute_input_json.MemberAsUInt("count", &ucnt);
        GOTO_IF_ERR(err, earlyexit);
        statistics->mutable_inference_stats()
            ->mutable_compute_input()
            ->set_count(ucnt);
        err = compute_input_json.MemberAsUInt("ns", &ucnt);
        GOTO_IF_ERR(err, earlyexit);
        statistics->mutable_inference_stats()->mutable_compute_input()->set_ns(
            ucnt);
      }

      {
        triton::common::TritonJson::Value compute_infer_json;
        err = infer_stats_json.MemberAsObject(
            "compute_infer", &compute_infer_json);
        GOTO_IF_ERR(err, earlyexit);

        err = compute_infer_json.MemberAsUInt("count", &ucnt);
        GOTO_IF_ERR(err, earlyexit);
        statistics->mutable_inference_stats()
            ->mutable_compute_infer()
            ->set_count(ucnt);
        err = compute_infer_json.MemberAsUInt("ns", &ucnt);
        GOTO_IF_ERR(err, earlyexit);
        statistics->mutable_inference_stats()->mutable_compute_infer()->set_ns(
            ucnt);
      }

      {
        triton::common::TritonJson::Value compute_output_json;
        err = infer_stats_json.MemberAsObject(
            "compute_output", &compute_output_json);
        GOTO_IF_ERR(err, earlyexit);

        err = compute_output_json.MemberAsUInt("count", &ucnt);
        GOTO_IF_ERR(err, earlyexit);
        statistics->mutable_inference_stats()
            ->mutable_compute_output()
            ->set_count(ucnt);
        err = compute_output_json.MemberAsUInt("ns", &ucnt);
        GOTO_IF_ERR(err, earlyexit);
        statistics->mutable_inference_stats()->mutable_compute_output()->set_ns(
            ucnt);
      }


      triton::common::TritonJson::Value batches_json;
      err = model_stat.MemberAsArray("batch_stats", &batches_json);
      GOTO_IF_ERR(err, earlyexit);

      for (size_t idx = 0; idx < batches_json.ArraySize(); ++idx) {
        triton::common::TritonJson::Value batch_stat;
        err = batches_json.IndexAsObject(idx, &batch_stat);
        GOTO_IF_ERR(err, earlyexit);

        auto batch_statistics = statistics->add_batch_stats();

        uint64_t ucnt;
        err = batch_stat.MemberAsUInt("batch_size", &ucnt);
        GOTO_IF_ERR(err, earlyexit);
        batch_statistics->set_batch_size(ucnt);

        {
          triton::common::TritonJson::Value compute_input_json;
          err = batch_stat.MemberAsObject("compute_input", &compute_input_json);
          GOTO_IF_ERR(err, earlyexit);

          err = compute_input_json.MemberAsUInt("count", &ucnt);
          GOTO_IF_ERR(err, earlyexit);
          batch_statistics->mutable_compute_input()->set_count(ucnt);
          err = compute_input_json.MemberAsUInt("ns", &ucnt);
          GOTO_IF_ERR(err, earlyexit);
          batch_statistics->mutable_compute_input()->set_ns(ucnt);
        }

        {
          triton::common::TritonJson::Value compute_infer_json;
          err = batch_stat.MemberAsObject("compute_infer", &compute_infer_json);
          GOTO_IF_ERR(err, earlyexit);

          err = compute_infer_json.MemberAsUInt("count", &ucnt);
          GOTO_IF_ERR(err, earlyexit);
          batch_statistics->mutable_compute_infer()->set_count(ucnt);
          err = compute_infer_json.MemberAsUInt("ns", &ucnt);
          GOTO_IF_ERR(err, earlyexit);
          batch_statistics->mutable_compute_infer()->set_ns(ucnt);
        }

        {
          triton::common::TritonJson::Value compute_output_json;
          err =
              batch_stat.MemberAsObject("compute_output", &compute_output_json);
          GOTO_IF_ERR(err, earlyexit);

          err = compute_output_json.MemberAsUInt("count", &ucnt);
          GOTO_IF_ERR(err, earlyexit);
          batch_statistics->mutable_compute_output()->set_count(ucnt);
          err = compute_output_json.MemberAsUInt("ns", &ucnt);
          GOTO_IF_ERR(err, earlyexit);
          batch_statistics->mutable_compute_output()->set_ns(ucnt);
        }
      }
    }
  }

earlyexit:
  GrpcStatusUtil::Populate(&status, err);
  TRITONSERVER_ErrorDelete(err);
#else
  auto err = TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNAVAILABLE,
      "the server does not suppport model statistics");
  GrpcStatusUtil::Populate(&status, err);
  TRITONSERVER_ErrorDelete(err);
#endif

  return status;
}

::grpc::Status
InferenceServiceImpl::ModelInfer(
    ::grpc::ServerContext* context, const inference::ModelInferRequest* request,
    inference::ModelInferResponse* response)
{

    nvtxEventAttributes_t eventAttrib2;
    eventAttrib2.version = NVTX_VERSION;
    eventAttrib2.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib2.colorType = NVTX_COLOR_ARGB;
    eventAttrib2.color = 0xFF000000;
    eventAttrib2.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib2.message.ascii = "ModelInfer";
    nvtxRangeId_t id2 = nvtxRangeStartEx(&eventAttrib2);

  grpc::Status status;

  InferHandlerState state(tritonserver_.get());

  TRITONSERVER_Error* err = nullptr;
#ifdef TRITON_ENABLE_TRACING
  if ((state.trace_manager_ != nullptr) && (state.trace_id_ != 0)) {
    state.trace_manager_->CaptureTimestamp(
        state.trace_id_, TRITONSERVER_TRACE_LEVEL_MIN, "GRPC_WAITREAD_END");
  }
#endif  // TRITON_ENABLE_TRACING

  int64_t requested_model_version;
  if (err == nullptr) {
    err = GetModelVersionFromString(
        request->model_version(), &requested_model_version);
  }

  if (err == nullptr) {
    uint32_t txn_flags;
    err = TRITONSERVER_ServerModelTransactionProperties(
        tritonserver_.get(), request->model_name().c_str(),
        requested_model_version, &txn_flags, nullptr /* voidp */);
    if ((err == nullptr) && (txn_flags & TRITONSERVER_TXN_DECOUPLED) != 0) {
      err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "ModelInfer RPC doesn't support models with decoupled "
          "transaction policy");
    }
  }

  // Create the inference request which contains all the
  // input information needed for an inference.
  TRITONSERVER_InferenceRequest* irequest = nullptr;
  if (err == nullptr) {
    err = TRITONSERVER_InferenceRequestNew(
        &irequest, tritonserver_.get(), request->model_name().c_str(),
        requested_model_version);
  }

  if (err == nullptr) {
    err = SetInferenceRequestMetadata(irequest, *request);
  }

  // Will be used to hold the serialized data in case explicit string
  // tensors are present in the request.
  std::list<std::string> serialized_data;

  if (err == nullptr) {
    err = InferGRPCToInput(
        tritonserver_, shm_manager_, *request, &serialized_data, irequest);
  }

  if (err == nullptr) {
    err = InferAllocatorPayload<inference::ModelInferResponse>(
        tritonserver_, shm_manager_, *request, std::move(serialized_data),
        response, &state.alloc_payload_);
  }

  if (err == nullptr) {
    err = TRITONSERVER_InferenceRequestSetReleaseCallback(
        irequest, InferRequestComplete, nullptr /* request_release_userp */);
  }
  if (err == nullptr) {
    err = TRITONSERVER_InferenceRequestSetResponseCallback(
        irequest, allocator_,
        &state.alloc_payload_ /* response_allocator_userp */,
        InferResponseComplete, reinterpret_cast<void*>(&state));
  }

  if (err == nullptr) {
    TRITONSERVER_InferenceTrace* trace = nullptr;
#ifdef TRITON_ENABLE_TRACING
    trace = state.trace_;
#endif  // TRITON_ENABLE_TRACING
    state.complete_ = false;
    state.response_ = response;
    err = TRITONSERVER_ServerInferAsync(tritonserver_.get(), irequest, trace);
  }

  if (err == nullptr) {
    // Wait until all callbacks are invoked
    {
      std::unique_lock<std::mutex> lk(state.mutex_);
      state.cv_.wait(lk, [&]() { return state.complete_; });
    }
    err = state.err_;
  }

  GrpcStatusUtil::Populate(&status, err);

  nvtxRangeEnd(id2);

  //std::cout << "Response Size: " << response->ByteSizeLong() << " bytes" << std::endl;
  return status;
}


void
InferenceServiceImpl::InferResponseComplete(
    TRITONSERVER_InferenceResponse* iresponse, const uint32_t flags,
    void* userp)
{
  InferHandlerState* state = reinterpret_cast<InferHandlerState*>(userp);

  // Defer to the callback with the final response
  if ((flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) == 0) {
    LOG_ERROR << "[INTERNAL] ModelInfer received a response without FINAL flag";
    return;
  }

  state->err_ = nullptr;
  // This callback is expected to be called exactly once for each request.
  // Will use the single response object in the response list to hold the
  // information.
  inference::ModelInferResponse* response = state->response_;

  if (iresponse == nullptr) {
    state->err_ = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, "received an unexpected null response");
  } else {
    state->err_ = InferResponseCompleteCommon(
        state->tritonserver_, iresponse, *response, state->alloc_payload_);
  }

  if (state->err_ != nullptr) {
    response->Clear();
  }

  LOG_TRITONSERVER_ERROR(
      TRITONSERVER_InferenceResponseDelete(iresponse),
      "deleting GRPC inference response");

#ifdef TRITON_ENABLE_TRACING
  if ((state->trace_manager_ != nullptr) && (state->trace_id_ != 0)) {
    state->trace_manager_->CaptureTimestamp(
        state->trace_id_, TRITONSERVER_TRACE_LEVEL_MIN, "GRPC_SEND_START");
  }
#endif  // TRITON_ENABLE_TRACING

  state->complete_ = true;
  state->cv_.notify_one();
}

TRITONSERVER_Error*
InferenceServiceImpl::InferResponseCompleteCommon(
    TRITONSERVER_Server* server, TRITONSERVER_InferenceResponse* iresponse,
    inference::ModelInferResponse& response,
    const AllocPayload<inference::ModelInferResponse>& alloc_payload)
{
  RETURN_IF_ERR(TRITONSERVER_InferenceResponseError(iresponse));

  const char *model_name, *id;
  int64_t model_version;
  RETURN_IF_ERR(TRITONSERVER_InferenceResponseModel(
      iresponse, &model_name, &model_version));
  RETURN_IF_ERR(TRITONSERVER_InferenceResponseId(iresponse, &id));

  response.set_id(id);
  response.set_model_name(model_name);
  response.set_model_version(std::to_string(model_version));

  // Propagate response parameters.
  uint32_t parameter_count;
  RETURN_IF_ERR(TRITONSERVER_InferenceResponseParameterCount(
      iresponse, &parameter_count));
  for (uint32_t pidx = 0; pidx < parameter_count; ++pidx) {
    const char* name;
    TRITONSERVER_ParameterType type;
    const void* vvalue;
    RETURN_IF_ERR(TRITONSERVER_InferenceResponseParameter(
        iresponse, pidx, &name, &type, &vvalue));
    inference::InferParameter& param = (*response.mutable_parameters())[name];
    switch (type) {
      case TRITONSERVER_PARAMETER_BOOL:
        param.set_bool_param(*(reinterpret_cast<const bool*>(vvalue)));
        break;
      case TRITONSERVER_PARAMETER_INT:
        param.set_int64_param(*(reinterpret_cast<const int64_t*>(vvalue)));
        break;
      case TRITONSERVER_PARAMETER_STRING:
        param.set_string_param(reinterpret_cast<const char*>(vvalue));
        break;
    }
  }

  // Go through each response output and transfer information to the
  // corresponding GRPC response output.
  uint32_t output_count;
  RETURN_IF_ERR(
      TRITONSERVER_InferenceResponseOutputCount(iresponse, &output_count));
  if (output_count != (uint32_t)response.outputs_size()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, "response output count mismatch");
  }

  for (uint32_t output_idx = 0; output_idx < output_count; ++output_idx) {
    const char* cname;
    TRITONSERVER_DataType datatype;
    const int64_t* shape;
    uint64_t dim_count;
    const void* base;
    size_t byte_size;
    TRITONSERVER_MemoryType memory_type;
    int64_t memory_type_id;
    void* userp;

    RETURN_IF_ERR(TRITONSERVER_InferenceResponseOutput(
        iresponse, output_idx, &cname, &datatype, &shape, &dim_count, &base,
        &byte_size, &memory_type, &memory_type_id, &userp));

    const std::string name(cname);

    // There are usually very few outputs so fastest just to look for
    // the one we want... could create a map for cases where there are
    // a large number of outputs. Or rely on order to be same...
    inference::ModelInferResponse::InferOutputTensor* output = nullptr;
    for (auto& io : *(response.mutable_outputs())) {
      if (io.name() == name) {
        output = &io;
        break;
      }
    }

    if (output == nullptr) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          "unable to find expected response output");
    }

    // If this output was requested as classification then remove the
    // raw output from the response and instead return classification
    // results as a string tensor
    const auto itr = alloc_payload.classification_map_.find(name);
    if (itr == alloc_payload.classification_map_.end()) {
      // Not classification...
      output->set_datatype(TRITONSERVER_DataTypeString(datatype));
      for (size_t idx = 0; idx < dim_count; idx++) {
        output->add_shape(shape[idx]);
      }
    } else {
      // Classification
      const uint32_t classification_count = itr->second;

      // For classification need to determine the batch size, if any,
      // because need to use that to break up the response for each
      // batch entry.
      uint32_t batch_size = 0;

      uint32_t batch_flags;
      RETURN_IF_ERR(TRITONSERVER_ServerModelBatchProperties(
          server, model_name, model_version, &batch_flags,
          nullptr /* voidp */));
      if ((dim_count > 0) &&
          ((batch_flags & TRITONSERVER_BATCH_FIRST_DIM) != 0)) {
        batch_size = shape[0];
      }

      // Determine the batch1 byte size of the tensor... needed when
      // the response tensor batch-size > 1 so that we know how to
      // stride though the tensor data.
      size_t batch1_element_count = 1;
      for (size_t idx = ((batch_size == 0) ? 0 : 1); idx < dim_count; idx++) {
        batch1_element_count *= shape[idx];
      }

      const size_t batch1_byte_size =
          batch1_element_count * TRITONSERVER_DataTypeByteSize(datatype);

      // Create the classification contents
      std::string serialized;

      size_t class_offset = 0;
      for (uint32_t bs = 0; bs < std::max((uint32_t)1, batch_size); ++bs) {
        std::vector<std::string> class_strs;
        RETURN_IF_ERR(TopkClassifications(
            iresponse, output_idx,
            reinterpret_cast<const char*>(base) + class_offset,
            ((class_offset + batch1_byte_size) > byte_size) ? 0
                                                            : batch1_byte_size,
            datatype, classification_count, &class_strs));

        // Serialize for binary representation...
        for (const auto& str : class_strs) {
          uint32_t len = str.size();
          serialized.append(reinterpret_cast<const char*>(&len), sizeof(len));
          if (len > 0) {
            serialized.append(str);
          }
        }

        class_offset += batch1_byte_size;
      }

      // Update the output with new datatype, shape and contents.
      output->set_datatype(
          TRITONSERVER_DataTypeString(TRITONSERVER_TYPE_BYTES));

      if (batch_size > 0) {
        output->add_shape(batch_size);
      }
      output->add_shape(
          std::min(classification_count, (uint32_t)batch1_element_count));

      (*response.mutable_raw_output_contents())[output_idx] =
          std::move(serialized);
    }
  }

  // Make sure response doesn't exceed GRPC limits.
  if (response.ByteSizeLong() > MAX_GRPC_MESSAGE_SIZE) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string(
            "Response has byte size " +
            std::to_string(response.ByteSizeLong()) +
            " which exceeds gRPC's byte size limit " + std::to_string(INT_MAX) +
            ".")
            .c_str());
  }

  return nullptr;  // success
}


}}  // namespace nvidia::inferenceserver