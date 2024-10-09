def torch_to_numpy(x):
    if x.device == "cpu":
        x = x.numpy()
    else:
        x = x.cpu().numpy()
    return x
