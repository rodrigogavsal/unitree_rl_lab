// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <cstring>

namespace isaaclab {

class Algorithms {
public:
    virtual std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) = 0;
    virtual void reset_state() {}

    std::vector<float> get_action() {
        std::lock_guard<std::mutex> lock(act_mtx_);
        return action;
    }
    
    std::vector<float> action;
protected:
    std::mutex act_mtx_;
};

class OrtRunner : public Algorithms {
public:
    void reset_state() override {
        for (auto& pair : hidden_states) {
            std::fill(pair.second.begin(), pair.second.end(), 0.0f);
        }
        std::cout << "[INFO] Memoria de la Red reseteada" << std::endl;
    }

    OrtRunner(std::string model_path) {
        // Init Model
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_model");
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        session_options.SetIntraOpNumThreads(1);
        session_options.SetInterOpNumThreads(1);
        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

        // 1. Escanear todas las ENTRADAS del modelo (obs, h_in, c_in...)
        for (size_t i = 0; i < session->GetInputCount(); ++i) {
            Ort::TypeInfo input_type = session->GetInputTypeInfo(i);
            auto shape = input_type.GetTensorTypeAndShapeInfo().GetShape();
            input_shapes.push_back(shape);
            
            size_t size = 1;
            for (const auto& dim : shape) {
                size *= dim;
            }
            input_sizes.push_back(size);

            auto input_name = session->GetInputNameAllocated(i, allocator);
            std::string name_str = input_name.get();
            input_names.push_back(input_name.release());

            // --- MAGIA RECURRENTE: Si el modelo pide memoria, la inicializamos a 0 ---
            if (name_str == "h_in" || name_str == "c_in") {
                hidden_states[name_str] = std::vector<float>(size, 0.0f);
            }
        }

        // 2. Escanear todas las SALIDAS del modelo (actions, h_out, c_out...)
        for (size_t i = 0; i < session->GetOutputCount(); ++i) {
            Ort::TypeInfo output_type = session->GetOutputTypeInfo(i);
            auto shape = output_type.GetTensorTypeAndShapeInfo().GetShape();
            output_shapes.push_back(shape);

            auto output_name = session->GetOutputNameAllocated(i, allocator);
            std::string name_str = output_name.get();
            output_names.push_back(output_name.release());

            // Dimensionamos el vector de acciones de la clase base
            if (name_str == "actions") {
                action.resize(shape[1]);
            }
        }
    }

    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // 3. Crear los tensores de entrada mezclando las observaciones y la memoria interna
        std::vector<Ort::Value> input_tensors;
        for(size_t i = 0; i < input_names.size(); ++i) {
            const std::string name_str(input_names[i]);
            float* data_ptr = nullptr;

            // ¿Es una observación que viene del robot?
            if (obs.find(name_str) != obs.end()) {
                data_ptr = obs.at(name_str).data();
            } 
            // ¿Es la memoria oculta (estado) de la red neuronal?
            else if (hidden_states.find(name_str) != hidden_states.end()) {
                data_ptr = hidden_states.at(name_str).data();
            } 
            else {
                throw std::runtime_error("Falta la entrada requerida por el ONNX: " + name_str);
            }

            auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, data_ptr, input_sizes[i], input_shapes[i].data(), input_shapes[i].size());
            input_tensors.push_back(std::move(input_tensor));
        }

        // 4. Ejecutar Inferencia (Paso de la red neuronal)
        auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), output_names.size());

        // 5. Repartir los resultados
        std::lock_guard<std::mutex> lock(act_mtx_);
        for (size_t i = 0; i < output_names.size(); ++i) {
            std::string out_name(output_names[i]);
            auto floatarr = output_tensors[i].GetTensorMutableData<float>();

            if (out_name == "actions") {
                // Guardar las acciones físicas para enviarlas a los motores
                std::memcpy(action.data(), floatarr, action.size() * sizeof(float));
            } 
            else if (out_name == "h_out") {
                // MAGIA RECURRENTE: Actualizar la memoria con el nuevo estado (h_out -> h_in)
                std::memcpy(hidden_states["h_in"].data(), floatarr, hidden_states["h_in"].size() * sizeof(float));
            }
            else if (out_name == "c_out") {
                // Por si en el futuro decides usar LSTM en lugar de GRU
                std::memcpy(hidden_states["c_in"].data(), floatarr, hidden_states["c_in"].size() * sizeof(float));
            }
        }
        return action;
    }

private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<int64_t> input_sizes;
    
    std::vector<std::vector<int64_t>> output_shapes; // Ahora soportamos múltiples formas de salida

    // Diccionario para guardar el estado oculto (memoria) de las RNN
    std::unordered_map<std::string, std::vector<float>> hidden_states;
};

}; // namespace isaaclab