import numpy as np
import os

def ddpg(
    env,
    agent,
    n_episodes=1000,
    window=100,
    max_t=300,
    checkpoint_dir=None,
    best_model_name=None,
    final_model_name=None,
):
    """Deep Deterministic Policy Gradient.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        window (int): window size for computing average score
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Get checkpoint paths
    best_model_path = os.path.join(
        checkpoint_dir, 
        best_model_name
    )
    final_model_path = os.path.join(
        checkpoint_dir, 
        final_model_name
    )
    
    # list containing scores from each episode
    scores = []
    actor_losses = []
    critic_losses = []
    max_avg_score = -np.inf
    best_result = {
        'params': {},
        'final_titer': -np.inf,
        'episode': 0,
        'predictions': None,
        'time_points': np.arange(env.config.TOTAL_DAYS)  # Time points for plotting
    }
    
    for i_episode in range(1, n_episodes + 1):
        
        # get init state
        state, _ = env.reset()
        agent.reset()
        
        # Store trajectory information
        states = []
        actions = []
        
        # run each episode
        total_reward = 0
        total_critic_loss = 0
        total_actor_loss = 0
        for t in range(max_t):
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            loss = agent.step(state, action, reward, next_state, done)
            
            # Store trajectory
            states.append(state[:4])  # Only store VCD, Glucose, Lactate, Titer
            actions.append(action)
            
            state = next_state
            total_reward += reward
            if done:
                break
            
            if loss:
                critic_loss, actor_loss = loss
                total_critic_loss += critic_loss.item()
                total_actor_loss += actor_loss.item()
        
        # save current episode total score    
        scores.append(total_reward)
        critic_losses.append(total_critic_loss)
        actor_losses.append(total_actor_loss)

        # Update best result if we have a better score
        final_titer = state[3]  # Titer is at index 3 in state
        if final_titer > best_result['final_titer']:
            best_result['final_titer'] = final_titer
            best_result['episode'] = i_episode
            best_result['params'] = {
                'feed_start': env.current_params['feed_start'],
                'feed_end': env.current_params['feed_end'],
                'Glc_feed_rate': np.mean([a[0] for a in actions]),  # Average feed rate
                'Glc_0': env.current_params['Glc_0'],
                'VCD_0': env.current_params['VCD_0']
            }
            # Store predictions for plotting
            best_result['predictions'] = np.array(states)

        # Print episode stats
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores)), end="")
        if len(scores) > window:
            avg_score = np.mean(scores[-window:])
            if avg_score > max_avg_score:
                max_avg_score = avg_score
                agent.save(best_model_path)

        if i_episode % window == 0:
            print(
                "\rEpisode {}/{} | Max Average Score: {:.2f}".format(
                    i_episode, n_episodes, max_avg_score
                ),
            )

    # Save final model
    agent.save(final_model_path)
    
    # Print optimal process conditions
    print("\nOptimal process conditions:")
    for param, value in best_result['params'].items():
        print(f"{param}: {value:.4f}")
    print(f"Predicted final titer: {best_result['final_titer']:.2f} mg/L")
    print(f"Found in episode: {best_result['episode']}")
    
    return best_result, (scores, actor_losses, critic_losses)