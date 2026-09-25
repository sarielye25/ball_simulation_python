# training settings
optimizer_name = 'adam'
learning_rate = 0.001
weight_decay = 0
batch_size = 128
max_updates = 10000
loss_name = 'mse' # does that directly select the corresponding loss function?
loss_reduction = 'mean' # the same question as above
shuffle = True # does that automatically shuffle the elements between batches?
drop_last = False # what does that mean?
device = 'cpu'
dtype = 'float32'
seed = 42 # how does that affect the training process?

# stopping and evaluation settings
stopping_ruler = "intermediate" # is this grammatically correct?
target_pass_rate =  0.95
patience_updates = 500
min_delta = 0.001
early_eval_updates = (0, 10, 20, 50, 100)
per_row_error_updates = (0, 10)
eval_interval = 100 # does that directly set up the evaluation schedule?
breakaway_band_N = 0.2 # what does this parameter mean?
tolerances = {
    "coarse": (0.1, 0.1),
    "intermediate": (0.01, 0.01),
    "fine": (0.001, 0.001)
}

# data and output settings
#data_dir =   to be decided
#run_dir =   to be decided
expected_train_rows = 8000
expected_validation_rows = 1000
expected_test_rows = 1000

scale_epsilon = 1e-8
near_zero_scale_policy = "replace_with_one"

input_columns = (
      "v0_m_s",
      "force_N",
      "force_duration_s",
      "observation_time_s",
  )

target_columns = (
    "displacement_m",
    "v_final_m_s",
  )
