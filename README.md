### Requirements
This program makes use of `nlopt`, `boost` nad `eigen` C++ libraries. Its parallel implementation makes use of `gnu_parallel`.

## USAGE
First compile running `python build.py`. To get the optimal values of $a_1$ and $a_2$, run `./bobyqa.out`. On the other hand, to run in parallel for an array of $\sigma_d$, execute the following command

```bash
parallel -j n --progress python parallel.py ::: {0..99}
```
where $n$ is the number of dedicated jobs.
