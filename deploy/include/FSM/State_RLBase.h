// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"

class State_RLBase : public FSMState {
public:
    State_RLBase(int state_mode, std::string state_string);
    
    void enter() {
        // set gain
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i) {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }

        env->robot->update();
        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            
            if (env->alg) { env->alg->reset_state(); }
            env->reset();
            auto sleepTill = clock::now() + dt;
            
            while (policy_thread_running){
                env->step();
                auto now = clock::now();
            if (now >= sleepTill) {
                // Si la IA tardó demasiado, reseteamos la línea temporal
                // para evitar el "bucle de la muerte".
                sleepTill = now + dt; 
            } else {
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
                }
            }
        });
    }

    void run();
    
    void exit() {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread;
    bool policy_thread_running = false;
};

REGISTER_FSM(State_RLBase)
