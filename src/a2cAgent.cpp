#include <torch/torch.h>

#include <iostream>
#include <chrono>  // for high_resolution_clock
using namespace torch;
using namespace std;
#include "structs.h"
#include <random>
#include <math.h>

class A2CAgent
{
  public:
    Policy policy_mu = Policy(3,1,false);
    Policy policy_sigma = Policy(3, 1, true);
    ValueFn value_fn = ValueFn(3,1);
    //at::Tensor dist; , policy_sigma
    //Model model = Model(4, 1);

    at::Tensor mu, sigma, value;
    //double sigma = 5;

    A2CAgent()
    {}

    ~A2CAgent()
    {}

    torch::Tensor act(torch::Tensor state)
    {
          mu = policy_mu.forward(state);
          sigma = policy_sigma.forward(state);
          value = value_fn.forward(state);

          auto action = torch::normal(mu.item<double>(), sigma.item<double>(), {1,1});
          action.set_requires_grad(true);
          action.requires_grad_(true);

          return action;
    }

    void learn(torch::Tensor state, 
               torch::Tensor next_state, 
               double reward,
               torch::Tensor action,
               torch::optim::Adam mu_optim,
               torch::optim::Adam sigma_optim,
               torch::optim::Adam value_optim,
               double gamma = 0.9
               )
    {
        value_optim.zero_grad();
        torch::Tensor next_state_value = value_fn.forward(next_state);
        torch::Tensor value_loss = torch::pow((reward + gamma*next_state_value) - value, 2);
        value_loss.backward();
        value_optim.step();

        mu_optim.zero_grad();
        sigma_optim.zero_grad();
        torch::Tensor p1 = - ( torch::pow((mu - action) ,2) / (torch::clamp(2*sigma, 1e-3)) );
        torch::Tensor p2 = - torch::log(std::sqrt(2 * M_PI) * sigma);
        torch::Tensor loss = p1 + p2;

        torch::Tensor entropy_loss =  1e-4 * (-(torch::log(2*M_PI*sigma) + 1)/2);
        torch::Tensor adv = (reward + gamma*next_state_value.detach()) - value.detach();
        loss = -loss*adv;

        torch::Tensor loss_tot = loss + entropy_loss ;
        loss_tot.backward();
        /*for (auto& p : policy_mu.parameters())
        {
            p.data().div_(torch::norm(p.data()));
        }*/
        mu_optim.step();
        /*for (auto& p : policy_sigma.parameters())
        {
            p.data().div_(torch::norm(p.data()));
        }*/
        sigma_optim.step();

        /*for (auto& p : policy_sigma.parameters())
        {
            //std::cout << "----- p grad " << p.data()<< std::endl;
            p.data().div_(torch::norm(p.data()));
            //std::cout << "----- p grad " << p.data() << std::endl;
            *//**//*auto tmp = 0.001*p.grad();
            p = p.detach();
            p -= tmp;*//**//*
        }*/
    }
};