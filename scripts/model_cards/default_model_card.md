---
{{ card_data }}
---


# Cloudcasting

## Model Description

<!-- Provide a longer summary of what this model is/does. -->
This model is trained to predict future frames of satellite data from past frames. It takes 3 hours
of recent satellkite imagery at 15 minute intervals and predicts 3 hours into the future also at
15 minute intervals. The satellite inputs and predictions are multispectral with 11 channels.


See [1] for the repo used to train the model.

- **Developed by:** Open Climate Fix and the Alan Turing Institute
- **License:** mit


# Training Details

## Data

<!-- This should link to a Data Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->
This was trained on EUMETSAT satellite imagery derived from the data stored in [this google public 
dataset](https://console.cloud.google.com/marketplace/product/bigquery-public-data/eumetsat-seviri-rss?hl=en-GB&inv=1&invt=AbniZA&project=solar-pv-nowcasting&pli=1). 

The data was processed using the protocol in [2]



## Results

The training logs for the current model can be found here:
- {{ wandb_link }}


### Software

- [1] https://github.com/openclimatefix/sat_pred
- [2] https://github.com/alan-turing-institute/cloudcasting