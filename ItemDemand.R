library(tidymodels)
library(embed)
library(ggplot2)
library(recipes)
library(vroom)
library(themis)
library(discrim)
library(keras)
library(tensorflow)
library(tune)
library(dials)
library(workflows)
library(gridExtra)
library(forecast)
library(timetk)
library(vroom)
library(modeltime)


train <- vroom("C:/Users/sfolk/Desktop/STAT348/Demand/train.csv")
test <- vroom("C:/Users/sfolk/Desktop/STAT348/Demand/test.csv")


storeItem <- train |> 
  filter(store == 4, item == 10)

storeItem2 <- train |> 
  filter(store == 3, item == 5)

# Time series plots
p1 <- storeItem %>%
  ggplot(aes(x = date, y = sales)) +
  geom_line() +
  geom_smooth(se = FALSE) +
  theme_minimal() +
  theme(axis.title = element_blank())

p2 <- storeItem2 %>%
  ggplot(aes(x = date, y = sales)) +
  geom_line() +
  geom_smooth(se = FALSE) +
  theme_minimal() +
  theme(axis.title = element_blank())

# ACF plots using forecast::ggAcf
p3 <- storeItem %>%
  pull(sales) %>%
  forecast::ggAcf(lag.max = 30) +
  theme_minimal() +
  theme(axis.title = element_blank())

p4 <- storeItem2 %>%
  pull(sales) %>%
  forecast::ggAcf(lag.max = 30) +
  theme_minimal() +
  theme(axis.title = element_blank())

p5 <- storeItem %>%
  pull(sales) %>%
  forecast::ggAcf(lag.max = 2*365) +
  theme_minimal() +
  theme(axis.title = element_blank())

p6 <- storeItem2 %>%
  pull(sales) %>%
  forecast::ggAcf(lag.max = 2*365) +
  theme_minimal() +
  theme(axis.title = element_blank())

# Arrange plots in 2x3 grid
panel_plot <- grid.arrange(p1, p3, p5,
                           p2, p4, p6,
                           nrow = 2, ncol = 3)

# Save plot
ggsave("C:/Users/sfolk/Downloads/time_series_panel.png", panel_plot, width = 15, height = 10)

my_recipe <- recipe(sales ~ ., data = storeItem) %>%
  step_date(date, features = c("dow", "month", "doy", "year")) %>%
  step_range(date_doy, min=0, max=pi) %>%
  step_mutate(sinDOY=sin(date_doy), cosDOY=cos(date_doy)) |>
  step_mutate(year_numeric = as.numeric(date_year)) |> 
  step_rm(date)


rf_model <- rand_forest(
  trees = tune(),      
  min_n = tune()        
) %>%
  set_mode("regression") %>%
  set_engine("ranger")


rf_workflow <- workflow() |> 
  add_model(rf_model) |> 
  add_recipe(my_recipe) 
  

storeItem_folds <- vfold_cv(storeItem, v = 5)


rf_grid <- grid_regular(
  trees(range = c(100, 1000)),
  min_n(range = c(2, 20)),
  levels = 5
)


rf_tune <- rf_workflow %>%
  tune_grid(
    resamples = storeItem_folds,
    grid = rf_grid,
    metrics = metric_set(smape)
  )

best_rf <- rf_tune %>%
  show_best(metric = "smape")

best_rf

ggsave("C:/Users/sfolk/Downloads/sales_plot.png", p)






create_recipe <- function(data) {
  recipe(sales ~ ., data = data) %>%
    # First remove store and item since they're constant
    step_rm(store, item) %>%
    # Then process date features
    step_date(date, features = c("dow", "month", "doy")) %>%
    step_range(date_doy, min = 0, max = pi) %>%
    step_mutate(
      sinDOY = sin(date_doy), 
      cosDOY = cos(date_doy),
      year_numeric = as.numeric(format(date, "%Y"))
    ) %>%
    step_rm(date_doy) %>%
    # Normalize remaining numeric predictors
    step_normalize(all_numeric_predictors()) %>%
    step_zv(all_predictors())
}

# Define ARIMA model
arima_model <- arima_reg(
  seasonal_period = 7,
  non_seasonal_ar = 5,
  non_seasonal_ma = 5,
  seasonal_ar = 2,
  seasonal_ma = 2,
  non_seasonal_differences = 2,
  seasonal_differences = 2
) %>%
  set_engine("auto_arima")

# First store-item combination (store 4, item 10)
storeItem1_train <- train %>% 
  filter(store == 4, item == 10)
storeItem1_test <- test %>% 
  filter(store == 4, item == 10)

# Second store-item combination (store 3, item 5)
storeItem2_train <- train %>% 
  filter(store == 3, item == 5)
storeItem2_test <- test %>% 
  filter(store == 3, item == 5)

# Create CV splits
cv_split1 <- time_series_split(
  storeItem1_train,
  assess = "3 months",
  initial = "2 years",
  cumulative = TRUE
)

cv_split2 <- time_series_split(
  storeItem2_train,
  assess = "3 months",
  initial = "2 years",
  cumulative = TRUE
)

# Fit models for first combination
arima_wf1 <- workflow() %>%
  add_recipe(create_recipe(storeItem1_train)) %>%
  add_model(arima_model) %>%
  fit(data = training(cv_split1))

cv_results1 <- modeltime_calibrate(arima_wf1,
                                   new_data = testing(cv_split1))

fullfit1 <- cv_results1 %>%
  modeltime_refit(data = storeItem1_train)

# Fit models for second combination
arima_wf2 <- workflow() %>%
  add_recipe(create_recipe(storeItem2_train)) %>%
  add_model(arima_model) %>%
  fit(data = training(cv_split2))

cv_results2 <- modeltime_calibrate(arima_wf2,
                                   new_data = testing(cv_split2))

fullfit2 <- cv_results2 %>%
  modeltime_refit(data = storeItem2_train)

# Create the four plots
p1 <- cv_results1 %>%
  modeltime_forecast(
    new_data = testing(cv_split1),
    actual_data = training(cv_split1)
  ) %>%
  plot_modeltime_forecast(
    .interactive = FALSE
  )

p2 <- cv_results2 %>%
  modeltime_forecast(
    new_data = testing(cv_split2),
    actual_data = training(cv_split2)
  ) %>%
  plot_modeltime_forecast(
    .interactive = FALSE
  )

p3 <- fullfit1 %>%
  modeltime_forecast(
    new_data = storeItem1_test,
    actual_data = storeItem1_train
  ) %>%
  plot_modeltime_forecast(
    .interactive = FALSE
  )

p4 <- fullfit2 %>%
  modeltime_forecast(
    new_data = storeItem2_test,
    actual_data = storeItem2_train
  ) %>%
  plot_modeltime_forecast(
    .interactive = FALSE
  )

# Combine all plots
myplot <- plotly::subplot(p1, p2, p3, p4, nrows = 2, titleX = TRUE, titleY = TRUE)

htmlwidgets::saveWidget(myplot, "C:/Users/sfolk/Downloads/forecast_plot.html")








fulltrain <- vroom("C:/Users/sfolk/Desktop/STAT348/Demand/train.csv")
fulltest <- vroom("C:/Users/sfolk/Desktop/STAT348/Demand/test.csv")

train <- fulltrain %>% filter(store == 4, item == 10)
test <- fulltest %>% filter(store == 4, item == 10)

cv_split <- time_series_split(train, assess="3 months", cumulative=TRUE)
cv_split %>%  tk_time_series_cv_plan() %>% 
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)

prophet_model <- prophet_reg() %>% 
  set_engine('prophet') %>% 
  fit(sales ~ date, data=training(cv_split))

cv_results <- modeltime_calibrate(prophet_model, new_data = testing(cv_split))

cv_results %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = FALSE
  )

p1 <- cv_results %>% modeltime_forecast(new_data = testing(cv_split),
                                        actual_data = training(cv_split)) %>% 
  plot_modeltime_forecast(.interactive = FALSE, .title = 'CV Preds: Store 4 Item 10')

full_fit <- cv_results %>% 
  modeltime_refit(data = train)

p2 <- full_fit %>% 
  modeltime_forecast(new_data = test,
                     actual_data = train) %>% 
  plot_modeltime_forecast(.interactive=FALSE, .title = '3 Month Forecast: Store 4 Item 10')

## Store-Item Combo 2:

train <- fulltrain %>% filter(store == 3, item == 5)
test <- fulltest %>% filter(store == 3, item == 5)

cv_split <- time_series_split(train, assess="3 months", cumulative=TRUE)
cv_split %>%  tk_time_series_cv_plan() %>% 
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)

prophet_model <- prophet_reg() %>% 
  set_engine('prophet') %>% 
  fit(sales ~ date, data=training(cv_split))

cv_results <- modeltime_calibrate(prophet_model, new_data = testing(cv_split))

cv_results %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = FALSE
  )

p3 <- cv_results %>% modeltime_forecast(new_data = testing(cv_split),
                                        actual_data = training(cv_split)) %>% 
  plot_modeltime_forecast(.interactive = FALSE, .title = 'CV Preds: Store 3 Item 5')

full_fit <- cv_results %>% 
  modeltime_refit(data = train)

p4 <- full_fit %>% 
  modeltime_forecast(new_data = test,
                     actual_data = train) %>% 
  plot_modeltime_forecast(.interactive=FALSE, .title = '3 Month Forecast: Store 3 Item 5')

myplot <- plotly::subplot(p1, p2, p3, p4, nrows = 2, titleX = TRUE, titleY = TRUE)

htmlwidgets::saveWidget(myplot, "C:/Users/sfolk/Downloads/prophet_plot.html")

