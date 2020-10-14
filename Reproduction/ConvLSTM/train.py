import model

available_ids = [i for i in range(0, num_trailers)]

from random import shuffle
shuffle(available_ids)

final_train_id = int(len(available_ids)*0.8)
train_ids = available_ids[:final_train_id]
val_ids = available_ids[final_train_id:]

# fit the model
history = model.fit_generator(
    generate_arrays(train_ids)
    , steps_per_epoch = len(train_ids)
  
    , validation_data = generate_arrays(val_ids)
    , validation_steps = len(val_ids)
  
    , epochs = 100
    , verbose = 1
    , shuffle = False
    , initial_epoch = 0
    )