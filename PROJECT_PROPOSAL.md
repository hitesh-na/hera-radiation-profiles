# Radiation Profiles & Pfotzer Peak Estimation

**Objective:** Map continuous atmospheric radiation curves to precisely locate the Pfotzer-Regener maximum and generate standardized profiles for cross-flight comparison.

The project will use CosmicWatch counts and environmental measurements from HERA balloon flights to construct count-rate-versus-altitude profiles. XGBoost regression will estimate detector CPS from altitude, pressure and other recorded sensor inputs. The primary products will be defensible profiles, explicit peak identification or unresolved statuses, and standardized comparisons across flights.

The prior STAR work studied feature importance with within-flight Random Forest and conventional Gradient Boosting fits. The present study keeps XGBoost as its sole predictive model family and tests generalization by withholding complete flights. A high within-flight R2 alone will not establish peak-localization success.

Data preparation will verify time units, detector deadtime/exposure, synchronization, sensor units and altitude coverage. Ascent and descent will be analyzed separately. Measured and predicted profiles will use a shared altitude grid without extrapolating beyond observations. An interior maximum must have sufficient supported coverage and a meaningful decline on both sides.

Six XGBoost variants will test the value of sensor features, feature engineering and hyperparameter tuning. An outer leave-one-flight-out design will isolate test flights. Inner grouped validation will select hyperparameters only among the remaining flights, and repeated seeds will quantify algorithm variability. Evaluation will report row RMSE/MAE/R2, binned profile errors, peak status agreement and peak-altitude differences only when both estimates pass screening.

The final study will include code, an executable notebook, source and exclusion records, full tuning histories, profile and peak tables, figures and a documented interpretation. Conditional counting errors, binning sensitivity, detector differences and the limited number of independent flights will be distinguished from genuine peak-location uncertainty.

**Current progress:** software is implemented and tested using explicitly synthetic fixtures. Actual HERA measurement files were absent from the provided materials, so all empirical conclusions remain pending. The next milestone is raw-data validation, followed by frozen-protocol real-data training and supervisor review of candidate peaks.
