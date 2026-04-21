# Speaker Notes by Slide

## Timing target

- Maximum allowed video length for a group of two: **14 minutes**
- Recommended spoken duration: **12:45 to 13:10**
- Suggested balance:
  - **Jordan**: around **6 min 20 s**
  - **Gustavo**: around **6 min 25 s**
- Final slide: leave **1 to 2 seconds** of silence, no narration needed

## Speaker split

- **Jordan**: Slides **1-11**, **22**, **24**
- **Gustavo**: Slides **12-21**, **23**
- **Slide 25**: no one speaks

## Slide-by-slide script

### Slide 1 -- Title
**Speaker:** Jordan  
**Time:** 10-12 s

Hello, we are Jordan Ikukele Bomolo Mia and Gustavo Garzo Chumilla, and in this mini-project we worked on snow pole detection for winter driving scenes from Trondheim.

### Slide 2 -- Outline
**Speaker:** Jordan  
**Time:** 10-12 s

I will first introduce the problem and the exploratory data analysis, and then Gustavo will cover the models, results, discussion, and the final conclusions.

### Slide 3 -- Background & Motivation
**Speaker:** Jordan  
**Time:** 30-35 s

The motivation for this task comes from winter driving. In snowy conditions, standard visual cues such as lane markings and road boundaries become much less reliable, so snow poles become an important reference for road localisation. From a computer vision perspective, this is a single-class object detection problem, but it is still challenging because the poles are thin, visually small, and often far away from the camera.

### Slide 4 -- Background & Motivation -- Visual Examples
**Speaker:** Jordan  
**Time:** 30-35 s

This slide gives a quick visual impression of the data. On the left, the `Road_poles_iPhone` examples are brighter and cleaner, with clearer snowy scenes. On the right, the `roadpoles_v1` examples are darker and look more difficult in terms of visibility and contrast. So even before looking at any statistics, we can already see that the two benchmarks are related, but not equally easy.

### Slide 5 -- Approach / Strategy
**Speaker:** Jordan  
**Time:** 30-35 s

Because of that, our strategy was not to jump directly into model training. We first treated the project as a dataset understanding problem. The goal was to analyse both leaderboards separately, understand what makes the task difficult, and then use those observations to justify model and training decisions later in the presentation.

### Slide 6 -- Approach / Strategy -- Key Reasoning
**Speaker:** Jordan  
**Time:** 30-35 s

More specifically, the EDA was designed to answer three questions. First, are these two datasets really similar enough to be interpreted in the same way? Second, is the main difficulty caused by object density, by object scale, by geometry, or by scene structure? And third, which properties of the data are most likely to explain the final leaderboard performance?

### Slide 7 -- Data Analysis -- Dataset Overview
**Speaker:** Jordan  
**Time:** 45-50 s

According to the bar chart on the left, the first major difference is dataset size. `Road_poles_iPhone` is much larger, with `942` training images, `261` validation images, and `138` test images. In contrast, `roadpoles_v1` contains `322` training images, `92` validation images, and `46` test images. So the iPhone dataset offers roughly three times more training data. That matters because, all else equal, we should expect more stable training and validation behaviour on the larger dataset. Also, both test sets have hidden labels, so all of our dataset analysis had to rely on the training and validation splits.

### Slide 8 -- Data Analysis -- Object Geometry
**Speaker:** Jordan  
**Time:** 55-60 s

This figure is one of the most informative ones in the EDA. It summarizes three geometric properties of the training set. The first bar plot shows median box area as a percentage of the image, the second shows the mean height-to-width ratio, and the third shows the mean number of boxes per image. The key point here is that `roadpoles_v1` is geometrically harder. Its poles are smaller on average and much more elongated, which we can see from the clearly larger height-to-width ratio. At the same time, the rightmost plot shows that both datasets are sparse, at around `1.2` poles per image. So this is not a crowding problem. The challenge is to detect very thin roadside objects accurately.

### Slide 9 -- Data Analysis -- Object Scale
**Speaker:** Jordan  
**Time:** 45-50 s

This histogram reinforces the same conclusion from a different angle. Most bounding boxes are concentrated very close to zero on the x-axis, which means the poles occupy only a tiny fraction of the image area. In other words, both datasets are dominated by small-object detection. The distribution for `roadpoles_v1` is also slightly shifted toward even smaller objects, which again supports the idea that it is the harder benchmark. This is exactly why image resolution becomes such an important design variable later on.

### Slide 10 -- Data Analysis -- Spatial Distribution
**Speaker:** Jordan  
**Time:** 35-40 s

The heatmaps show where pole centres appear in normalized image coordinates. In both datasets, the detections are concentrated in a narrow horizontal band instead of being spread everywhere. That makes sense physically, because the poles are positioned along the road and therefore appear at fairly structured locations in the frame. So this is not a random-object dataset; it is a structured driving-scene dataset.

### Slide 11 -- Data Analysis -- Spatial Distribution Insights
**Speaker:** Jordan  
**Time:** 25-30 s

The practical consequence is that model failures are less likely to come from clutter or object overlap, and more likely to come from visibility and localisation issues on small, thin poles. So with that dataset analysis in mind, we can now motivate the specific model choices and training setups used in the project.

### Slide 12 -- Overview -- Two Architectures
**Speaker:** Gustavo  
**Time:** 20-25 s

To study the task from two different perspectives, we used two detection families. First, YOLO as a lightweight single-stage baseline. Second, Faster R-CNN as a two-stage detector with an FPN-based Torchvision pipeline. Importantly, both model families were used on both datasets, not one model per leaderboard.

### Slide 13 -- Model 1 -- YOLO Baseline Architecture
**Speaker:** Gustavo  
**Time:** 35-40 s

Starting with YOLO, we used it as the fast baseline detector. The main reason is simplicity: it is easy to train, easy to export, and naturally relevant for edge deployment because it predicts in a single forward pass. In the notebooks, the baseline setup uses `yolov8n`, so the idea here was not to maximize capacity, but to establish a lightweight reference point before moving to the stronger two-stage detector.

### Slide 14 -- Model 1 -- YOLO Training Config
**Speaker:** Gustavo  
**Time:** 20-25 s

This slide shows the actual baseline configuration from the notebooks. We used `yolov8n`, trained for `100` epochs at image size `640`. So this part of the project was intentionally simple. It gave us a quick baseline for both datasets, but the strongest completed and validated runs later came from Faster R-CNN.

### Slide 15 -- Model 2 -- Faster R-CNN Architecture
**Speaker:** Gustavo  
**Time:** 35-40 s

For the second model, we used Faster R-CNN, which follows a two-stage pipeline. First, the Region Proposal Network suggests candidate object locations. Then, the ROI head classifies and refines those proposals. This is useful here because the poles are thin and often far away, so careful localisation matters a lot. In practice, this model family gave us the best validated results on both datasets.

### Slide 16 -- Model 2 -- Faster R-CNN Training Config
**Speaker:** Gustavo  
**Time:** 30-35 s

The most important implementation detail for Faster R-CNN was actually the dataset pipeline. Our labels were originally in YOLO format, which uses normalized centre coordinates and box size, but Torchvision's detector expects absolute corner coordinates. So, as shown in the code snippet, we had to convert every box from normalized YOLO format into pixel-space coordinates. In the notebooks, the Faster R-CNN training also uses `AdamW`, a cosine annealing schedule, and `100` epochs.

### Slide 17 -- Results -- Validation Set Metrics
**Speaker:** Gustavo  
**Time:** 40-45 s

This table now focuses on the strongest completed validation runs, which are the Faster R-CNN runs on both datasets. The main point is that the same model performs clearly better on `Road_poles_iPhone` than on `roadpoles_v1`. On the iPhone dataset, Faster R-CNN reaches `0.7796` at `mAP@0.5:0.95`, while on `roadpoles_v1` it reaches `0.6701`. Precision and recall follow the same pattern. So the results support the EDA very directly.

### Slide 18 -- Results -- Leaderboard
**Speaker:** Gustavo  
**Time:** 25-30 s

We also evaluated on the hidden test server. In the final submissions, Faster R-CNN was the stronger model on both leaderboards. The score is around `69%` on `Road poles iPhone` and around `66%` on `roadpoles_v1`. Both are lower than the validation results, which is expected, because the hidden test set introduces unseen conditions.

### Slide 19 -- Results -- Same Model on Two Datasets
**Speaker:** Gustavo  
**Time:** 30-35 s

This slide compares the same model family on two datasets, which makes the interpretation cleaner. The `mAP@50` values are almost identical, so Faster R-CNN is able to find poles in both cases. The real gap appears in `mAP@0.5:0.95`, where the difference is about `0.11`. That means the harder dataset is not stopping the detector from seeing the objects, but from localising them tightly.

### Slide 20 -- Discussion -- Why Faster R-CNN Performed Best
**Speaker:** Gustavo  
**Time:** 35-40 s

This discussion slide summarizes why Faster R-CNN ended up being the strongest validated model in our project. First, the two-stage pipeline helps with careful localisation of thin objects. Second, the FPN-based multi-scale representation is well aligned with poles at different distances. And third, the results are coherent with the EDA, because the same model behaves better on the easier dataset and worse on the harder one.

### Slide 21 -- Discussion -- Limitations & Future Work
**Speaker:** Gustavo  
**Time:** 30-35 s

This slide is important because it keeps the comparison realistic. Even though Faster R-CNN was our strongest validated model, `roadpoles_v1` is still clearly harder, and the lower hidden-test performance suggests domain shift in weather and lighting. Looking forward, the most promising improvements would be better hyperparameter sweeps for both model families, pseudo-labeling with unlabeled data, and test-time augmentation or ensembles.

### Slide 22 -- Key Learning Points
**Speaker:** Jordan  
**Time:** 40-45 s

For us, the main learning points are captured in these three blocks. First, the EDA was genuinely useful, because it predicted that `roadpoles_v1` would be harder, and the results later confirmed that. Second, building the Faster R-CNN pipeline gave us a better understanding of how two-stage detectors work in practice, especially proposal generation and box refinement. And third, a large part of the project was not just model selection, but engineering work: dataset formatting, YAML configuration, path handling, and adapting labels between frameworks.

### Slide 23 -- Sustainability -- Compute Energy
**Speaker:** Gustavo  
**Time:** 40-45 s

Finally, on sustainability, our estimate is based on training on an NVIDIA RTX 4090 with a TDP of `450` watts. Assuming around `4` hours of total compute time, that corresponds to approximately `1.8` kilowatt-hours. The slide also provides a simple comparison: at around `45` watts of laptop power draw, that would correspond to roughly `40` hours of coding. So for a project of this scale, especially one based on fine-tuning pretrained models rather than training from scratch, the footprint remains relatively modest.

### Slide 24 -- Summary
**Speaker:** Jordan  
**Time:** 30-35 s

To conclude, this table brings the main message together. The best validated results in our notebooks came from Faster R-CNN, and those results were stronger on `Road_poles_iPhone` than on `roadpoles_v1`. So our final takeaway is that Faster R-CNN was our best-performing model family, and that the EDA explained the dataset ranking quite well.

### Slide 25 -- Thank you for listening
**Speaker:** None  
**Time:** 0 s

No narration. Leave one or two seconds of silence so the video closes cleanly.
