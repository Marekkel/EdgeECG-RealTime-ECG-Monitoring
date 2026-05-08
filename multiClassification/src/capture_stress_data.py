
import numpy as np


X_test = np.load('/home/icsl/pi_eval_dataset/X_test.npy')
y_test = np.load('/home/icsl/pi_eval_dataset/y_test.npy')

idx_N = np.where(y_test == 0)[0] # assume 0 is N
idx_S = np.where(y_test == 1)[0] # assume 1 is S
idx_V = np.where(y_test == 2)[0] # assume 2 is V
idx_F = np.where(y_test == 3)[0] # assume 3 is F
idx_Q = np.where(y_test == 4)[0] # assume 4 is Q

# 3. extract all rare samples (S, V, F)
rare_indices = np.concatenate([idx_S, idx_V, idx_F])
rare_count = len(rare_indices)

# 4. randomly sample equal amounts of N and Q samples to create a 1:1 stress environment
np.random.shuffle(idx_N)
np.random.shuffle(idx_Q)
balanced_N = idx_N[:rare_count]
balanced_Q = idx_Q[:rare_count]

# 5. form new "balanced stress test set"
stress_test_idx = np.concatenate([rare_indices, balanced_N, balanced_Q])
np.random.shuffle(stress_test_idx) # shuffle the order

X_stress = X_test[stress_test_idx]
y_stress = y_test[stress_test_idx]

print(f"stress test set built successfully! Total samples: {len(y_stress)}")
print(f"rare samples containing rare abnormal heartbeats: {rare_count}")


np.save('/home/icsl/pi_eval_dataset/X_stress.npy', X_stress)
np.save('/home/icsl/pi_eval_dataset/y_stress.npy', y_stress)