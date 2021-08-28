from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import torch

def run(model, optimizer, loss_fn, max_epochs, train_generator, test_generator, writer, RUN_TIME, WILL_CHECK_TIMINGS, USE_AUTOCAST, SAVE_FILE_AT):
  scaler = GradScaler()

  torch.cuda.empty_cache()

  if (WILL_CHECK_TIMINGS):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    iteration_start = torch.cuda.Event(enable_timing=True)
    iteration_end = torch.cuda.Event(enable_timing=True)
    eval_function_start = torch.cuda.Event(enable_timing=True)
    eval_function_end = torch.cuda.Event(enable_timing=True)
    loss_function_start = torch.cuda.Event(enable_timing=True)
    loss_function_end = torch.cuda.Event(enable_timing=True)
    backprop_function_start = torch.cuda.Event(enable_timing=True)
    backprop_function_end = torch.cuda.Event(enable_timing=True)

    timing_accumulator = 0
    eval_accumulator = 0
    loss_accumulator = 0
    backprop_accumulator = 0
    start.record()

  iteration = 0
  for epoch in range(max_epochs):

    train_loss = 0
    test_loss = 0

    for local_batch, local_labels in train_generator:
        if (iteration_start):
            iteration_start.record()

        optimizer.zero_grad()        

        with autocast(enabled=USE_AUTOCAST==True):
            if (eval_function_start):            
                eval_function_start.record()
            local_labels_pred = model(local_batch)
            if (eval_function_end):
                eval_function_end.record()
                loss_function_start.record()
            train_loss = loss_fn(local_labels_pred, local_labels)
            if (loss_function_end):
                loss_function_end.record()
        del local_batch, local_labels, local_labels_pred

        if (backprop_function_start):
            backprop_function_start.record()

        if (USE_AUTOCAST==True):        
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            train_loss.backward()
            optimizer.step()
        
        if (backprop_function_end):
            backprop_function_end.record()

        #Validation ...
        if (iteration % 10 == 0):
            with torch.set_grad_enabled(False):
                for local_batch, local_labels in test_generator:
                     local_batch, local_labels = local_batch, local_labels

                     with autocast():
                         local_labels_pred = model(local_batch)
                         test_loss = loss_fn(local_labels_pred, local_labels)
                     del local_batch, local_labels, local_labels_pred
                     break
                writer.add_scalar('Loss/train', train_loss, iteration)
                writer.add_scalar('Loss/test', test_loss, iteration)
                print ("Iteration: " + str(iteration) + ", Remaining Time: " + str((RUN_TIME - timing_accumulator) / 1000) + " seconds")

        if (SAVE_FILE_AT and iteration % 500 == 0):
            torch.save(model.state_dict(), SAVE_FILE_AT)

        iteration += 1
        
        if (WILL_CHECK_TIMINGS):        
            iteration_end.record()
            torch.cuda.synchronize()

            timing_accumulator += iteration_start.elapsed_time(iteration_end)
            eval_accumulator += eval_function_start.elapsed_time(eval_function_end)
            loss_accumulator += loss_function_start.elapsed_time(loss_function_end)
            backprop_accumulator += backprop_function_start.elapsed_time(backprop_function_end)


        if (RUN_TIME and timing_accumulator >= RUN_TIME):
            break
    if (RUN_TIME and timing_accumulator >= RUN_TIME):
        break

  if (WILL_CHECK_TIMINGS):
    end.record()
  torch.cuda.empty_cache()

  if (WILL_CHECK_TIMINGS):
    print("Total time: " + str(start.elapsed_time(end) / 1000))
    print("Loop time: " + str(timing_accumulator / 1000))
    print("Model evaluation time: " + str(eval_accumulator / 1000))
    print("Loss function time: " + str(loss_accumulator / 1000))
    print("Backward pass: " + str(backprop_accumulator / 1000))
    print("Remaining time: " + str((timing_accumulator - (eval_accumulator + loss_accumulator + backprop_accumulator)) / 1000))

  print("Number of iterations: " + str(iteration))
