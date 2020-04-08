#include <torch/torch.h>
#include <iostream>
#include <ATen/ATen.h>

#include "a2cAgent.cpp"
#include "environment.cpp"

#include <random>
#include <iterator>
#include <algorithm>
#include "../proto/tensorboard_logger.h"

int main()
{
  auto agent = A2CAgent();

  const char* model_name = "../models/pendulum.xml";
  auto env = Environment(model_name, true);

  torch::optim::Adam mu_optim(agent.policy_mu.parameters(), torch::optim::AdamOptions(0.01));
  torch::optim::Adam value_optim(agent.value_fn.parameters(), torch::optim::AdamOptions(0.01));

  const int NUM_EPISODES = 1000;

  for (int i = 0; i < NUM_EPISODES; i++)
  {
  	// we start at angular position 0 and angular velocity 0
  	torch::Tensor start_state = torch::tensor({cos(0.0), sin(0.0), 0.0});
  	double episode_reward = 0.0;
  	double episode_length = 0.0;

  	std::string add = "";
  	while(true)
  	{
        double action = agent.act(start_state);
        if (i % 500 == 0)
        {
            if( i == 500) {
                agent.sigma /= 10;
                std::cout << agent.sigma << std::endl;
            }

            add = "TEST  ";
            auto next = env.step(true, action);
            double reward = std::get<0>(next);
            auto next_state = std::get<1>(next);

            episode_reward+=reward;

            double next_pos = std::get<0>(next_state);
            double next_vel = std::get<1>(next_state);
            torch::Tensor next_state_tensor = torch::tensor({cos(next_pos), sin(next_pos), next_vel});
            start_state = next_state_tensor;

            episode_length++;

            if (episode_length > 1)
                break;

        }
        else
        {
            add = "";
            //std::cout << "action " << action << std::endl;
            auto next = env.step(false, action);

            //std::cout << "reward " <<  std::get<0>(next) << std::endl;

            double reward = std::get<0>(next);
            auto next_state = std::get<1>(next);

            episode_reward+=reward;

            double next_pos = std::get<0>(next_state);
            double next_vel = std::get<1>(next_state);
            //std::cout << "next_pos " <<  std::get<0>(next_state) << std::endl;
            //std::cout << "next_vel " <<  std::get<1>(next_state) << std::endl;
            // next contains the next state and the reward
            torch::Tensor next_state_tensor = torch::tensor({cos(next_pos),sin(next_pos),next_vel});

            agent.learn(start_state, next_state_tensor, reward,
                        action, mu_optim, value_optim );

            start_state = next_state_tensor;

            episode_length++;

            if (episode_length > 1000)
                break;
        }
  	}
  	// print total reward for statistics
  	cout<< add << i << "   " << episode_reward << endl;
  	env.reset();
  }
  env.stop();

}
