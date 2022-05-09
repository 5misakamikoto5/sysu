import argparse
from wrappers import make_env
import gym
from argument import dqn_arguments, pg_arguments
from tqdm import tqdm


def parse():
    parser = argparse.ArgumentParser(description="SYSU_RL_HW2")
    parser.add_argument('--train_pg', default=False, type=bool, help='whether train policy gradient')
    parser.add_argument('--train_dqn', default=True, type=bool, help='whether train DQN')

    parser = dqn_arguments(parser)
    # parser = pg_arguments(parser)
    args = parser.parse_args()
    return args


def run(args):
    if args.train_pg:
        env_name = args.env_name
        env = gym.make(env_name)
        from agent_dir.agent_pg import AgentPG
        agent = AgentPG(env, args)
        agent.run()

    if args.train_dqn:
        env_name = args.env_name
        env = make_env(env_name)
        input_size = env.observation_space.shape[0]
        output_size = env.action_space.n
        # from agent_dir.agent_dqn import AgentDQN
        # agent = AgentDQN(env, args, input_size, output_size)
        # for i_episode in range(args.n_episodes): 
        #     obs = env.reset()
        #     episode_reward = 0
        #     done = False
        #     while not done:
        #         action = agent.make_action(obs)
        #         next_obs, reward, done, info = env.step(action)
        #         agent.store_transition = (obs, action, reward, next_obs, done)
        #         episode_reward += reward
        #         obs = next_obs
        #         if agent.buffer.__len__() >= args.buffer_size:
        #             agent.train()
        #     print(i_episode, "reward:", episode_reward)
        from agent_dir.agent_dqn import AgentDQN
        agent = AgentDQN(env, args, input_size, output_size)
        agent.run(env,args)
        


if __name__ == '__main__':
    args = parse()
    run(args)
