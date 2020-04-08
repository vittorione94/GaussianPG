#include <torch/torch.h>
#include <iostream>
#include <ATen/ATen.h>

#include "a2cAgent.cpp"
#include "environment.cpp"

#include <random>
#include <iterator>
#include <algorithm>
#include "../proto/tensorboard_logger.h"

void print(std::vector<int> const &input)
{
	for (int i = 0; i < input.size(); i++) {
		std::cout << input.at(i) << std::endl;
	}
}

int main() 
{
  auto agent = A2CAgent();
  /*for (const auto& p : agent.policy_mu.parameters())
  {
  	std::cout << p << std::endl;
  }*/

  const char* model_name = "../models/pendulum.xml";
  auto env = Environment(model_name, true);

  //GOOGLE_PROTOBUF_VERIFY_VERSION;
  //TensorBoardLogger logger("../log_dir/my_log2.pb");
  //logger.add_scalar("reward", 0, 1.0);
  // at::Tensor n = at::normal( 0.0, torch::tensor({10.0}) );

  //at::_shape_as_tensor(n)
  // std::cout << n << std::endl;

  torch::optim::Adam mu_optim(agent.policy_mu.parameters(), torch::optim::AdamOptions(0.01));
  //torch::optim::Adam sigma_optim(agent.policy_sigma.parameters(), torch::optim::AdamOptions(0.01));
  torch::optim::Adam value_optim(agent.value_fn.parameters(), torch::optim::AdamOptions(0.01));

  const int NUM_EPISODES = 1000;

  for (int i = 0; i < NUM_EPISODES; i++)
  {
  	// we start at angular position 0 and angular velocity 0
  	torch::Tensor start_state = torch::tensor({
                                                      /*asin(0.0),
                                                      acos(0.0),
                                                      atan(0.0),
                                                      tan(0.0),
                                                      std::pow(0.0,2),
                                                      0.0,*/
                                                      cos(0.0),
                                                      sin(0.0),
                                                      0.0});
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
            torch::Tensor next_state_tensor = torch::tensor({
                                                                    /*asin(next_pos),
                                                                    acos(next_pos),
                                                                    atan(next_pos),
                                                                    tan(next_pos),
                                                                    std::pow(next_pos,2),
                                                                    next_pos,*/
                                                                    cos(next_pos),
                                                                    sin(next_pos),
                                                                    next_vel});
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
            torch::Tensor next_state_tensor = torch::tensor({
                                                                    /*asin(next_pos),
                                                                    acos(next_pos),
                                                                    atan(next_pos),
                                                                    tan(next_pos),
                                                                    std::pow(next_pos,2),
                                                                    next_pos,*/
                                                                    cos(next_pos),
                                                                    sin(next_pos),
                                                                    next_vel});

            agent.learn(start_state,
                        next_state_tensor,
                        reward,
                        action,
                        //target_action,
                        //target_policy_grad,
                        //e_v,
                        //w,
                        //e_u1
                    //e_u2,
                    //e_u3
                    mu_optim,
                    //sigma_optim,
                    value_optim
            );

            /*for (const auto& p : agent.policy_mu.parameters())
            {
                std::cout << "----- " << p << std::endl;
            }*/

            start_state = next_state_tensor;

            episode_length++;

            if (episode_length > 1000)
                break;
        }
  	}


  	// write total reward and episode length for statistics
  	// logger.add_scalar("reward", 0, 1);
  	cout<< add << i << "   " << episode_reward << endl;
  	env.reset();
    //logger.add_scalar("length", i, episode_length);
  }


  /*for (int i = 0; i < 100; i++)
  {
  	env.step(true);
  }
  */
  env.stop();
  // we need first to create an environment
  /*
  NOTE: this is how we sample a vector
  std::vector<int> test, out;
  test.push_back(1);
  test.push_back(2);
  test.push_back(3);
  test.push_back(4);
  test.push_back(5);
  test.push_back(6);

  std::sample(test.begin(), test.end(), std::back_inserter(out),
                2, std::mt19937{std::random_device{}()});

  print(test);
  print(out);*/
}
