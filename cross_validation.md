Cross Validation
================

``` r
library(tidyverse)
```

    ## ── Attaching packages ───────────────────────────────────────────────────────────────── tidyverse 1.3.0 ──

    ## ✓ ggplot2 3.3.2     ✓ purrr   0.3.4
    ## ✓ tibble  3.0.3     ✓ dplyr   1.0.0
    ## ✓ tidyr   1.1.0     ✓ stringr 1.4.0
    ## ✓ readr   1.3.1     ✓ forcats 0.5.0

    ## ── Conflicts ──────────────────────────────────────────────────────────────────── tidyverse_conflicts() ──
    ## x dplyr::filter() masks stats::filter()
    ## x dplyr::lag()    masks stats::lag()

``` r
library(modelr)
library(mgcv)
```

    ## Loading required package: nlme

    ## 
    ## Attaching package: 'nlme'

    ## The following object is masked from 'package:dplyr':
    ## 
    ##     collapse

    ## This is mgcv 1.8-33. For overview type 'help("mgcv-package")'.

``` r
knitr::opts_chunk$set(
  fig.width = 6,
  fig.asp = 0.6,
  out.width = "90%"
)

theme_set(theme_minimal() + theme(legend.position = "bottom"))

options(
  ggplot2.continuous.colour = "viridis",
  ggplot2.continuous.fill = "viridis"
)

scale_colour_discrete = scale_colour_viridis_d
scale_fill_discrete = scale_fill_viridis_d
```

## Simulated dataset

``` r
nonlin_df = 
  tibble(
    id = 1:100,
    x = runif(100, 0, 1),
    y = 1 - 10 * (x - .3) ^ 2 + rnorm(100, 0, .3)
  )
```

Look at the data

``` r
nonlin_df %>% 
  ggplot(aes(x = x, y = y)) +
  geom_point()
```

<img src="cross_validation_files/figure-gfm/unnamed-chunk-2-1.png" width="90%" />

## Cross- validation by hand

Generate a training dataset and a testing dataset

``` r
train_df = sample_n(nonlin_df, size = 80)
test_df = anti_join(nonlin_df, train_df, by = "id")
```

Fit three models.

``` r
linear_model = lm(y ~x, data = train_df)
smooth_mod = gam(y ~ s(x), data = train_df)
wiggly_mod = gam(y ~ s(x, k = 30), sp = 10e-6, data = train_df)
```

Can I see what I just did ?

``` r
train_df %>%
  gather_predictions(linear_model, smooth_mod, wiggly_mod) %>% 
  ggplot(aes(x = x, y = y)) +
  geom_point() +
  geom_line(aes(y = pred), color = "red") + 
  facet_grid(. ~ model)
```

<img src="cross_validation_files/figure-gfm/unnamed-chunk-5-1.png" width="90%" />

The smooth model does best for prediction accuracy \!

Look at prediction accuracy - (computing the root mean sq for these
model) - use the testing dataset

``` r
rmse(linear_model, data = test_df)
```

    ## [1] 0.8319449

``` r
rmse(smooth_mod, data = test_df)
```

    ## [1] 0.3353538

``` r
rmse(wiggly_mod, data = test_df)
```

    ## [1] 0.3234575

This is what cross validation does - picks the smooth model to pick the
best model to make a prediction.

used when we are looking at unnested datasets.

## Cross validation using `modelr`

``` r
cv_df = 
  crossv_mc(nonlin_df, 100)
```

what is happening here ?

``` r
cv_df %>%  pull(train) %>% .[[1]] %>% as_tibble()
```

    ## # A tibble: 79 x 3
    ##       id      x      y
    ##    <int>  <dbl>  <dbl>
    ##  1     1 0.995  -3.85 
    ##  2     3 0.711  -0.629
    ##  3     4 0.409   1.21 
    ##  4     6 0.973  -3.83 
    ##  5     7 0.770  -1.13 
    ##  6     8 0.553   0.511
    ##  7     9 0.694  -0.973
    ##  8    11 0.731  -0.927
    ##  9    12 0.0958  0.389
    ## 10    13 0.127   0.771
    ## # … with 69 more rows

``` r
cv_df %>%  pull(test) %>% .[[1]] %>% as_tibble()
```

    ## # A tibble: 21 x 3
    ##       id      x      y
    ##    <int>  <dbl>  <dbl>
    ##  1     2 0.887  -2.45 
    ##  2     5 0.135   1.25 
    ##  3    10 0.0756  0.619
    ##  4    18 0.555  -0.605
    ##  5    24 0.149   0.583
    ##  6    25 0.209   1.10 
    ##  7    29 0.0438  0.489
    ##  8    31 1.00   -3.98 
    ##  9    32 0.104   0.291
    ## 10    33 0.220   0.729
    ## # … with 11 more rows

``` r
cv_df = 
  cv_df %>% 
  mutate(
    train = map(train, as_tibble),
    test = map(test, as_tibble)
  )
```

Let’s try to fit models and get RSME for them

``` r
cv_df = 
cv_df %>% 
  mutate(
    linear_model =  map(.x = train, ~lm(y ~ x, data = .x)),
    smooth_mod = map(.x = train, ~ gam(y ~ s(x), data = .x)),
    wiggly_mod = map(.x = train, ~gam(y ~ s(x, k = 30), sp = 10e-6, data = .x))
  ) %>% 
  mutate(
    rmse_linear = map2_dbl(.x  = linear_model, .y = test, ~rmse(model = .x, data = .y)),
    rmse_smooth = map2_dbl(.x  = smooth_mod, .y = test, ~rmse(model = .x, data = .y)),
    rmse_wiggly = map2_dbl(.x  = wiggly_mod, .y = test, ~rmse(model = .x, data = .y))
  )
```

Produce the RMSE that corresponds to each of these.

What do these results say about model choice

``` r
cv_df %>% 
  select(starts_with("rmse")) %>% 
  pivot_longer(
    everything(),
    names_to = "model",
    values_to = "rmse",
    names_prefix = "rmse_"
  ) %>% 
  ggplot(aes(x = model, y = rmse)) +
  geom_violin()
```

<img src="cross_validation_files/figure-gfm/unnamed-chunk-11-1.png" width="90%" />

Based on the distribution of the rmse and the plots - we can say that
the smooth model is doing the best.

compute averages…

``` r
cv_df %>% 
  select(starts_with("rmse")) %>% 
  pivot_longer(
    everything(),
    names_to = "model",
    values_to = "rmse",
    names_prefix = "rmse_"
  ) %>% 
  group_by(model) %>% 
  summarize(avg_rmse = mean(rmse))
```

    ## `summarise()` ungrouping output (override with `.groups` argument)

    ## # A tibble: 3 x 2
    ##   model  avg_rmse
    ##   <chr>     <dbl>
    ## 1 linear    0.825
    ## 2 smooth    0.280
    ## 3 wiggly    0.331

smooth is the lowest, hence does better than the other two \!
