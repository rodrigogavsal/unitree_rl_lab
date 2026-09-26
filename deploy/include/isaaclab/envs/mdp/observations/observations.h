// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "isaaclab/envs/manager_based_rl_env.h"

// dependencias para el puente de rclcpp con un thread secundario
#ifdef USE_ROS2_VISION
    #include <rclcpp/rclcpp.hpp>
    #include <std_msgs/msg/float32_multi_array.hpp>
    #include <mutex>
    #include <thread>
#endif

namespace isaaclab {
    namespace mdp {
        #ifdef USE_ROS2_VISION
        struct ROS2HeightSubscriber {
            std::vector<float> latest_heights;
            std::mutex mtx;
            rclcpp::Node::SharedPtr node;
            rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub;
            std::thread spin_thread;
            bool initialized = false;

            ROS2HeightSubscriber() : latest_heights(187, 0.0f) {}

            void init() {
                if (initialized) return;
                
                // 1. Inicializar rclcpp si el ejecutable principal no lo ha hecho
                if (!rclcpp::ok()) {
                    int argc = 0;
                    char **argv = nullptr;
                    rclcpp::init(argc, argv);
                }

                // 2. Crear el nodo y la suscripción
                node = rclcpp::Node::make_shared("ia_vision_subscriber");
                sub = node->create_subscription<std_msgs::msg::Float32MultiArray>(
                    "/ut/perception/height_vector", rclcpp::SensorDataQoS(),
                    [this](const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
                        std::lock_guard<std::mutex> lock(mtx);
                        if (msg->data.size() == 187) {
                            latest_heights = msg->data;
                        }
                    });

                // 3. Lanzar el hilo que actualiza los mensajes constantemente
                spin_thread = std::thread([this]() {
                    rclcpp::spin(node);
                });
                spin_thread.detach();
                
                initialized = true;
            }
        };

        // Instancia global única del escáner
        static ROS2HeightSubscriber g_height_subscriber;
        #endif

        REGISTER_OBSERVATION(base_ang_vel){
            auto & asset = env->robot;
            auto & data = asset->data.root_ang_vel_b;
            return std::vector<float>(data.data(), data.data() + data.size());
        }

        REGISTER_OBSERVATION(projected_gravity){
            auto & asset = env->robot;
            auto & data = asset->data.projected_gravity_b;
            return std::vector<float>(data.data(), data.data() + data.size());
        }

        REGISTER_OBSERVATION(joint_pos){
            auto & asset = env->robot;
            std::vector<float> data;

            std::vector<int> joint_ids;
            try {
                joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
            } catch(const std::exception& e) {
            }

            if(joint_ids.empty()){
                data.resize(asset->data.joint_pos.size());
                for(size_t i = 0; i < asset->data.joint_pos.size(); ++i)
                {
                    data[i] = asset->data.joint_pos[i];
                }
            }else{
                data.resize(joint_ids.size());
                for(size_t i = 0; i < joint_ids.size(); ++i){
                    data[i] = asset->data.joint_pos[joint_ids[i]];
                }
            }
            return data;
        }

        REGISTER_OBSERVATION(joint_pos_rel){
            auto & asset = env->robot;
            std::vector<float> data;

            data.resize(asset->data.joint_pos.size());
            for(size_t i = 0; i < asset->data.joint_pos.size(); ++i) {
                data[i] = asset->data.joint_pos[i] - asset->data.default_joint_pos[i];
            }

            try {
                std::vector<int> joint_ids;
                joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
                if(!joint_ids.empty()) {
                    std::vector<float> tmp_data;
                    tmp_data.resize(joint_ids.size());
                    for(size_t i = 0; i < joint_ids.size(); ++i){
                        tmp_data[i] = data[joint_ids[i]];
                    }
                    data = tmp_data;
                }
            } catch(const std::exception& e) {}
            return data;
        }

        REGISTER_OBSERVATION(joint_vel_rel){
            auto & asset = env->robot;
            auto data = asset->data.joint_vel;

            try {
                const std::vector<int> joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();

                if(!joint_ids.empty()) {
                    data.resize(joint_ids.size());
                    for(size_t i = 0; i < joint_ids.size(); ++i) {
                        data[i] = asset->data.joint_vel[joint_ids[i]];
                    }
                }
            } catch(const std::exception& e) {}
            return std::vector<float>(data.data(), data.data() + data.size());
        }

        REGISTER_OBSERVATION(last_action){
            auto data = env->action_manager->action();
            return std::vector<float>(data.data(), data.data() + data.size());
        };

        REGISTER_OBSERVATION(velocity_commands){
            std::vector<float> obs(3);
            auto & joystick = env->robot->data.joystick;

            const auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

            obs[0] = std::clamp(joystick->ly(), cfg["lin_vel_x"][0].as<float>(), cfg["lin_vel_x"][1].as<float>());
            obs[1] = std::clamp(-joystick->lx(), cfg["lin_vel_y"][0].as<float>(), cfg["lin_vel_y"][1].as<float>());
            obs[2] = std::clamp(-joystick->rx(), cfg["ang_vel_z"][0].as<float>(), cfg["ang_vel_z"][1].as<float>());

            return obs;
        }

        REGISTER_OBSERVATION(gait_phase){
            float period = params["period"].as<float>();
            float delta_phase = env->step_dt * (1.0f / period);

            env->global_phase += delta_phase;
            env->global_phase = std::fmod(env->global_phase, 1.0f);

            std::vector<float> obs(2);
            obs[0] = std::sin(env->global_phase * 2 * M_PI);
            obs[1] = std::cos(env->global_phase * 2 * M_PI);
            return obs;
        }

        #ifdef USE_ROS2_VISION
        REGISTER_OBSERVATION(height_scanner){
            // 1. Asegurarnos de que el nodo de visión está corriendo
            g_height_subscriber.init();

            // 2. Extraer el último mapa conocido a la velocidad de la CPU
            std::vector<float> obs;{
                std::lock_guard<std::mutex> lock(g_height_subscriber.mtx);
                obs = g_height_subscriber.latest_heights;
            }

            return obs;
            // std::vector<float> obs(187, -0.1f); // Fuerzas suelo liso
            // return obs;
        }
        #endif
    }
}