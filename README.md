# ml-compiler-optimzation

### Prepare input files

```sh
clang -O0 -Xclang -disable-O0-optnone -emit-llvm -S *.c
llvm-link *.ll -S -o <Linked_IR>.ll
```

### Generate XFG

```python
# Linked IRs are in <data_folder>/*/*.ll
i2v_prep.construct_xfg(data_folder) # produces <data_folder>/*_preprocessed/xfg/
```

