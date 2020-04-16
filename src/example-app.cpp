#include <torch/torch.h>
#include <iostream>
#include <ATen/ATen.h>

#include "a2cAgent.cpp"
#include "environment.cpp"

#include <random>
#include <iterator>
#include <algorithm>

int main(int argc, char** argv)
{
    A2CAgent agent = A2CAgent();

    const char* model_name;
    if (argc > 1) {
        model_name = argv[1];
    } else {
        model_name = "../models/pendulum.xml";
    }

    Environment env = Environment(model_name, true);

    torch::optim::Adam mu_optim(agent.policy_mu.parameters(), torch::optim::AdamOptions(0.001));
    torch::optim::Adam sigma_optim(agent.policy_sigma.parameters(), torch::optim::AdamOptions(0.001));
    torch::optim::Adam value_optim(agent.value_fn.parameters(), torch::optim::AdamOptions(0.01));

    const int NUM_EPISODES = 15000;
    double episode_reward, episode_length;

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis1(-M_PI, M_PI);
    std::uniform_real_distribution<> dis2(-2.0, 2.0);

    for (int i = 0; i < NUM_EPISODES; i++)
    {
        // we start at angular position RANDOM and angular velocity RANDOM
        double random_start_pos =  ( i % 100 == 0) ? 3.14: dis1(gen); //
        double random_start_vel =  ( i % 100 == 0) ? 0.0 : dis2(gen); //
        std::cout << random_start_pos << "   " << random_start_vel  <<std::endl;

        env.set_randomStart(random_start_pos, random_start_vel);
        //random_start_pos , 0.0
        torch::Tensor start_state = torch::tensor({cos(random_start_pos), sin(random_start_pos), random_start_vel});
        episode_reward = 0.0;
        episode_length = 0.0;

        bool render = (i % 500 == 0) ? true : false;
        int max_ep_len = (i % 500 == 0) ? 300 : 200;
        std::string add = "";
        while(true)
        {
            torch::Tensor action = agent.act(start_state);
            add = "";
            double act = std::clamp(action.item<double>(), -2.0, 2.0);
            auto next = env.step(render, act);
            double reward = std::get<0>(next);
            auto next_state = std::get<1>(next);

            episode_reward+=reward;

            double next_pos = std::get<0>(next_state);
            double next_vel = std::get<1>(next_state);

            // next contains the next state and the reward  next_pos next_acc
            torch::Tensor next_state_tensor = torch::tensor({cos(next_pos),sin(next_pos),next_vel});

            agent.learn(start_state, next_state_tensor, reward,
                        action, mu_optim, sigma_optim, value_optim );

            start_state = next_state_tensor;

            episode_length++;
            if (episode_length > max_ep_len)
                break;
        }
        // print total reward for statistics
        cout<< add << i << " ------------------------------>   " << episode_reward << endl;
        env.reset();
    }
    env.stop();

}