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

To study the task from two different perspectives, we used two detection families. First, YOLOv8s as a single-stage detector focused on speed and deployment. Second, Faster R-CNN with a ResNet-50 FPN backbone as a two-stage detector, which is slower but often strong on object detection benchmarks.

### Slide 13 -- Model 1 -- YOLOv8s Architecture
**Speaker:** Gustavo  
**Time:** 40-45 s

Starting with YOLOv8s, the motivation is shown directly on this slide. Because the poles occupy only about two percent of the image width, we need enough capacity to represent very small objects. That is why the small model was preferred over the nano version. The other reason is deployment: YOLO processes the whole image in one forward pass, which makes it attractive for real-time use on an edge device. On the right, the model summary highlights the final setup: one class, COCO pretraining, batch size eight, and input size `1280` by `1280`.

### Slide 14 -- Model 1 -- YOLOv8s Training Config
**Speaker:** Gustavo  
**Time:** 25-30 s

This slide shows the actual training configuration. The key points are `150` epochs, image size `1280`, patience `30`, and augmentations such as HSV color variation, horizontal flipping, mosaic, scaling, and translation. The main design idea here was not just to make the model larger, but to give it more usable pixels per pole.

### Slide 15 -- Model 2 -- Faster R-CNN Architecture
**Speaker:** Gustavo  
**Time:** 40-45 s

For the second model, we used Faster R-CNN, which follows a two-stage pipeline. First, the Region Proposal Network suggests candidate object locations. Then, the ROI head classifies and refines those proposals. The advantage of using a ResNet-50 FPN backbone is that feature maps are computed at multiple scales, which is relevant here because poles can appear at very different distances. So this model represents the more classical and more explicit multi-stage detection approach.

### Slide 16 -- Model 2 -- Faster R-CNN Training Config
**Speaker:** Gustavo  
**Time:** 30-35 s

The most important implementation detail for Faster R-CNN was actually the dataset pipeline. Our labels were originally in YOLO format, which uses normalized center coordinates and box size, but Torchvision’s detector expects absolute corner coordinates. So, as shown in the code snippet, we had to convert every box from normalized YOLO format into pixel-space coordinates. We also used train-time augmentation with color jitter and horizontal flipping.

### Slide 17 -- Results -- Validation Set Metrics
**Speaker:** Gustavo  
**Time:** 45-50 s

According to this table, both models performed very strongly at `mAP@50`, with values above `0.99`. So both detectors can usually find the poles. The main difference appears in the stricter metric `mAP@0.5:0.95`, where YOLO reaches `0.788` while Faster R-CNN reaches `0.626`. YOLO also achieves much higher recall, `0.976` compared with `0.679`. So overall, the table suggests that YOLO is more consistent and more precise across varying overlap thresholds.

### Slide 18 -- Results -- Leaderboard
**Speaker:** Gustavo  
**Time:** 25-30 s

We also evaluated on the hidden test server. On the `Road poles iPhone` leaderboard, YOLO achieved about `69%`, while on the `roadpoles_v1` leaderboard Faster R-CNN achieved about `66%`. Both are lower than the validation results, which is expected, because the hidden test set introduces conditions that the models did not directly tune on.

### Slide 19 -- Results -- mAP@50 vs mAP@0.5:0.95 Gap
**Speaker:** Gustavo  
**Time:** 35-40 s

This slide helps interpret why the stricter metric matters. On the left, we explain the difference between the two measures. On the right, the table and the gap values summarize it numerically. YOLO drops from `0.990` to `0.788`, while Faster R-CNN drops from `0.993` to `0.626`. So although both models detect the objects at IoU `0.5`, YOLO keeps much better performance when the localisation requirement becomes stricter. In practice, that means tighter boxes.

### Slide 20 -- Discussion -- Why YOLO Outperformed Faster R-CNN
**Speaker:** Gustavo  
**Time:** 35-40 s

This discussion slide summarizes the most likely reasons. First, dataset size: YOLO was trained on the larger benchmark. Second, resolution: the YOLO setup explicitly pushed image size higher, which is critical for small objects. And third, augmentation: mosaic and related augmentations increase visual diversity. So even though Faster R-CNN is theoretically attractive for small-object detection, in our setting YOLO had more favorable conditions.

### Slide 21 -- Discussion -- Limitations & Future Work
**Speaker:** Gustavo  
**Time:** 35-40 s

This slide is important because it keeps the comparison realistic. Faster R-CNN was trained on the smaller dataset, so the comparison is not fully balanced. The lower hidden-test performance also points to domain shift, especially weather and lighting variation. Looking forward, the most promising improvements would be increasing resolution for Faster R-CNN, using pseudo-labels from the unlabeled RoadPoles-MSJ data, and trying test-time augmentation or ensembles.

### Slide 22 -- Key Learning Points
**Speaker:** Jordan  
**Time:** 45-50 s

For us, the main learning points are captured in these three blocks. First, resolution matters a lot for small-object detection, which fits directly with the EDA findings. Second, building the Faster R-CNN pipeline gave us a better understanding of how two-stage detectors work in practice, especially proposal generation and box refinement. And third, a large part of the project was not just model selection, but engineering work: dataset formatting, YAML configuration, path handling, and adapting labels between frameworks.

### Slide 23 -- Sustainability -- Compute Energy
**Speaker:** Gustavo  
**Time:** 40-45 s

Finally, on sustainability, our estimate is based on training on an NVIDIA RTX 4090 with a TDP of `450` watts. Assuming around `4` hours of total compute time, that corresponds to approximately `1.8` kilowatt-hours. The slide also provides a simple comparison: at around `45` watts of laptop power draw, that would correspond to roughly `40` hours of coding. So for a project of this scale, especially one based on fine-tuning pretrained models rather than training from scratch, the footprint remains relatively modest.

### Slide 24 -- Summary
**Speaker:** Jordan  
**Time:** 35-40 s

To conclude, this table brings the main message together. Both models detect poles reliably at `mAP@50`, but YOLOv8s gives the better overall balance between localisation quality and deployability. So our final takeaway is that, for this task, the best practical model is YOLOv8s, and the most important modeling lever is preserving enough image resolution for very small objects.

### Slide 25 -- Thank you for listening
**Speaker:** None  
**Time:** 0 s

No narration. Leave one or two seconds of silence so the video closes cleanly.
