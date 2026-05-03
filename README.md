# Greenhouse Gas Emissions Predictor

This project builds an end-to-end machine learning pipeline for predicting corporate greenhouse gas emissions.

## What the Model Predicts

The model predicts a company's combined Scope 1 and Scope 2 location-based greenhouse gas emissions.

The prediction target is:

```text
scope1_plus_scope2_location_mt
```

In plain English, the model tries to answer:

```text
Given what we know about a company, what are its combined direct emissions and purchased-energy emissions?
```

## Scope 1 and Scope 2 Emissions

Scope 1 emissions are direct emissions from a company's own operations.

Examples include:

```text
fuel burned in company factories
company-owned trucks
onsite boilers or furnaces
industrial processes
```

Scope 2 emissions are indirect emissions from purchased energy.

Examples include:

```text
electricity bought from the grid
purchased steam
purchased heating or cooling
```

Together:

```text
Scope 1 + Scope 2 = direct company emissions + emissions from purchased energy
```

## Model Inputs

The model uses company attributes and emissions-related fields as inputs:

```text
reporting_year
scope1_mt_co2e
scope2_location_mt_co2e
emissions_intensity_mt_per_musd
revenue_usd_millions
net_zero_target_year
country
region
sector
industry
third_party_verified
```

The model output is one numeric prediction:

```text
predicted combined Scope 1 + Scope 2 emissions
```

## Current Modeling Caveat

This is currently a lifecycle-focused MLOps project, not a perfect predictive modeling problem.

The model includes `scope1_mt_co2e` and `scope2_location_mt_co2e` as inputs while predicting `scope1_plus_scope2_location_mt`. Because the target is closely related to those input columns, the model can achieve strong metrics partly because the prediction task is relatively easy.

That is acceptable for this first project because the goal is to learn the full ML lifecycle:

```text
data versioning -> data processing -> model training -> experiment tracking -> API serving -> containerization
```
