import d4rl
import gym
import pandas as pd
import pyrallis
from matplotlib import pyplot as plt

from cql import ReplayBuffer, FullyConnectedQFunction, TanhGaussianPolicy, ContinuousCQL, TrainConfig, modify_reward, \
	compute_mean_std, \
	normalize_states, wrap_env, set_seed, eval_actor, extract_actor_param, dataset_split, wandb_init, dataset_split_4, dataset_split_4_16, dataset_split_diy
import torch
import numpy as np
from dataclasses import asdict, dataclass
import wandb
import argparse

def combine_agents(main_meta_learner, meta_learners):
    for i in range(len(meta_learners)):
        for main_param, agent_param in zip(main_meta_learner.critic_1.parameters(), meta_learners[i].critic_1.parameters()):
            if(i == 0):
                main_param.data.copy_(agent_param)
            else:
                main_param.data.copy_(main_param * (i/(i+1)) + agent_param * (1/(i+1)))


        for main_param, agent_param in zip(main_meta_learner.critic_2.parameters(), meta_learners[i].critic_2.parameters()):
            if(i == 0):
                main_param.data.copy_(agent_param)
            else:
                main_param.data.copy_(main_param * (i/(i+1)) + agent_param * (1/(i+1)))


        for main_param, agent_param in zip(main_meta_learner.actor.parameters(),meta_learners[i].actor.parameters()):
            if (i == 0):
                main_param.data.copy_(agent_param)
            else:
                main_param.data.copy_(main_param * (i / (i + 1)) + agent_param * (1 / (i + 1)))


    return main_meta_learner

def distribute_agents(main_agent, agents):
    for i in range(len(agents)):
        for main_agent_param, agent_param in zip(main_agent.critic_1.parameters(), agents[i].critic_1.parameters()):
            agent_param.data.copy_(main_agent_param)

        for main_agent_param, agent_param in zip(main_agent.critic_2.parameters(), agents[i].critic_2.parameters()):
            agent_param.data.copy_(main_agent_param)

        for main_agent_param, agent_param in zip(main_agent.actor.parameters(), agents[i].actor.parameters()):
            agent_param.data.copy_(main_agent_param)

    return agents

class cql_learner:
	def __init__(
		self,
		env,
		state_dim,
		action_dim,
		max_action,
		dataset,
		config,
	) -> None:
		self.env = env
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.max_action = max_action
		self.dataset = dataset
		self.config = config
		self.replay_buffer = ReplayBuffer(
			state_dim,
			action_dim,
			config.buffer_size,
			config.device,
		)
		self.critic_1 = FullyConnectedQFunction(
			state_dim,
			action_dim,
			config.orthogonal_init,
			config.q_n_hidden_layers,
		).to(config.device)

		self.critic_2 = FullyConnectedQFunction(
			state_dim,
			action_dim,
			config.orthogonal_init
		).to(config.device
		)
		self.critic_1_optimizer = torch.optim.Adam(list(self.critic_1.parameters()), config.qf_lr)
		self.critic_2_optimizer = torch.optim.Adam(list(self.critic_2.parameters()), config.qf_lr)
		self.actor = TanhGaussianPolicy(
			state_dim,
			action_dim,
			max_action,
			log_std_multiplier=config.policy_log_std_multiplier,
			orthogonal_init=config.orthogonal_init,
		).to(config.device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), config.policy_lr)

		kwargs = {
			"critic_1": self.critic_1,
			"critic_2": self.critic_2,
			"critic_1_optimizer": self.critic_1_optimizer,
			"critic_2_optimizer": self.critic_2_optimizer,
			"actor": self.actor,
			"actor_optimizer": self.actor_optimizer,
			"discount": config.discount,
			"soft_target_update_rate": config.soft_target_update_rate,
			"device": config.device,
			# CQL
			"target_entropy": -np.prod(env.action_space.shape).item(),
			"alpha_multiplier": config.alpha_multiplier,
			"use_automatic_entropy_tuning": config.use_automatic_entropy_tuning,
			"backup_entropy": config.backup_entropy,
			"policy_lr": config.policy_lr,
			"qf_lr": config.qf_lr,
			"bc_steps": config.bc_steps,
			"target_update_period": config.target_update_period,
			"cql_n_actions": config.cql_n_actions,
			"cql_importance_sample": config.cql_importance_sample,
			"cql_lagrange": config.cql_lagrange,
			"cql_target_action_gap": config.cql_target_action_gap,
			"cql_temp": config.cql_temp,
			"cql_alpha": config.cql_alpha,
			"cql_max_target_backup": config.cql_max_target_backup,
			"cql_clip_diff_min": config.cql_clip_diff_min,
			"cql_clip_diff_max": config.cql_clip_diff_max,
		}
		self.trainer = ContinuousCQL(**kwargs)

@pyrallis.wrap()
def train(num_worker, config: TrainConfig):
	num_agents = num_worker
	env = gym.make(config.env)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]

	dataset = d4rl.qlearning_dataset(env)
	max_action = float(env.action_space.high[0])
	if config.normalize_reward:
		modify_reward(
			dataset,
			config.env,
			reward_scale=config.reward_scale,
			reward_bias=config.reward_bias,
		)

	if config.normalize:
		state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
	else:
		state_mean, state_std = 0, 1

	dataset["observations"] = normalize_states(
		dataset["observations"], state_mean, state_std
	)
	dataset["next_observations"] = normalize_states(
		dataset["next_observations"], state_mean, state_std
	)
	env = wrap_env(env, state_mean=state_mean, state_std=state_std)
	seed = config.seed
	set_seed(seed, env)
	dicts = dataset_split_diy(dataset, 1000*num_agents, num_agents)
	agents = [cql_learner(env, state_dim, action_dim, max_action, dict, config) for dict in dicts]
	agent_center = cql_learner(env, state_dim, action_dim, max_action, dataset, config)
	print("---------------------------------------")
	print(f"Training CQL, Env: {config.env}, Seed: {seed}")
	print("---------------------------------------")
	wandb_init(asdict(config))
	# evaluations_1 = []
	# evaluations_2 = []
	evaluations_center = []
	steps = []
	step = 0
	for t in range(int(config.max_timesteps)):
		center_params = extract_actor_param(agent_center)
		for (i, agent) in enumerate(agents):
			if t == 0:
				agent.replay_buffer.load_d4rl_dataset(agent.dataset)
			batch = agent.replay_buffer.sample(config.batch_size)
			batch = [b.to(config.device) for b in batch]
			log_dict = agent.trainer.train(batch, center_params)
                  
		if (t + 1) % 50 == 0:
			agent_center = combine_agents(agent_center, agents)
			agents = distribute_agents(agent_center, agents)
			if (t + 1) % config.eval_freq == 0:
				step += 500
				steps.append(step)
				print(f"Time steps: {t + 1}")
				eval_scores = eval_actor(
					env,
					agent_center.actor,
					device=config.device,
					n_episodes=config.n_episodes,
					seed=config.seed,
				)
				eval_score = eval_scores.mean()
				normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
				evaluations_center.append(normalized_eval_score)
				wandb.log(
					{"server_normalized_score": normalized_eval_score},
					step=t,
				)
				print("---------------------------------------")
				print(
					f"Evaluation over {config.n_episodes} episodes: "
					f"{eval_score:.3f} , agent_center  D4RL score: {normalized_eval_score:.3f}"
				)

def test(config, path):
    env = gym.make(config.env)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    dataset = d4rl.qlearning_dataset(env)
    max_action = float(env.action_space.high[0])
    
    agent = cql_learner(env, state_dim, action_dim, max_action, dataset, config)
    
    agent = torch.load(path)

    eval_scores = eval_actor(
		env,
		agent.actor,
		device=config.device,
		n_episodes=config.n_episodes,
		seed=config.seed,
	)
    
    eval_score = eval_scores.mean()
    normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
    print(normalized_eval_score)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='halfcheetah-medium-expert-v2', help='the name of environment')
    parser.add_argument('--method', type=str, default='DRPO', help='method name')
    parser.add_argument('--batch_size', type=int, default=20, help='batch_size')
    parser.add_argument('--local_update', type=int, default=20, help='frequency of local update')
    parser.add_argument('--num_worker', type=int, default=10, help='number of federated agents')


    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--lr_a', type=float, default=1e-4, help='learning rate of actor')
    parser.add_argument('--lr_c', type=float, default=1e-4, help='learning rate of critic')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--actor_path', type=str, default='./actor.pkl', help='path of actor model')

    args = parser.parse_args()
    config = TrainConfig()
    config.qf_lr = args.lr_c
    config.policy_lr = args.lr_a
    if args.mode == 'train':
        train(args.num_worker, config)
    else:
    	test(config, args.actor_path)