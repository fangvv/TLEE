

def lr_schedule(config, lr, optimizer, epoch):
    schedule = config.model.schedule
    if (epoch + 1) in schedule:
        lr *= 0.1
        for param_group in optimizer.param_groups:  #在每次更新参数前迭代更改学习率 
            param_group["lr"] = lr       
