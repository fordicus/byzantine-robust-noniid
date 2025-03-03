# Dependency

- torch 2.5.1
- torchvision 0.20.1
- scipy 1.13.1
- numpy 1.26.4
- matplotlib 3.9.2

# Code Organization

- `exp3.py`:  Main Script Example
- `codes/`:   Subroutines
- `outputs/`: Experiment Outputs

# Example Command

```
python exp3.py \\  
  --use-cuda \\  
  -n 25 \\  
  -f 12 \\  
  --noniid \\  
  --bucketing 0.0 \\  
  --attack 'ALIE' \\  
  --agg 'byro' \\  
  --momentum 0.9 \\  
  --val_set_sz 100 \\  
  --client_delta 0.0 \\  
  --target '1111111111' \\  
  --seed 0 \\  
  --identifier 'all' \\  
  --log_interval 1 \\  
  --dispersion_idea 0 \\  
  --bucketing_idea 0
```

# Command Flags

- `--seed`:					(integer) random number generators, e.g., {0, 1, 2}
- `--identifier`:			(string)  subfolder name for outputs
- `--plot`:					(boolean) see `if not args.plot` in `exp3.py`
- `-n`:						(integer) the total number of clients
- `-f`:						(integer) the number of Byzantine clients
- `--attack`:				(string)  {"mimic", "IPM", "ALIE", "BF", "LF"}
- `--agg`:					(string)  {"byro", "cp", "rfa", "cm", "krum"}
- `--noniid`:				(boolean) necessary
- `--bucketing_idea`:		(integer) use 0 to stick to the random-shuffling
- `--bucketing`:			(float)   0.0 means no bucketing, 2.0 means two in each bucket
- `--momentum`:				(float)   0.9 is beta-emphasis on the memory not stoch. grad.
- `--clip-tau`:				(float)   manual clipping radius for "cp", e.g., 10
- `--clip-scaling`:			(string)  if specified "linear", see Karimireddy et al. 2022
- `--val_set_sz`:			(integer) how many validation data points to be collected
- `--train_mb_sz`:			(integer) per-client mini-batch size, e.g., 32 data points
- `--client_delta`:			(float)   intermittent Byzantine disruption, similar to delta [0.0, 0.5)
- `--target`:				(string)  10-digit binary, `1111111111` means all clasess from 10, sparse definitions select subclasses
- `--dispersion_idea`:		(integer) 0 means the standard definition of attacks by adding no noise
- `--dispersion_factor`:	(float)   convex combination between [0.0, 1.0], 0.0 means some small noise

# License

This repo is covered under [The MIT License](LICENSE).
