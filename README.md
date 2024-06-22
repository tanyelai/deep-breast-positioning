
# Mammographic Breast Positioning Assessment via Deep Learning
Breast cancer is a primary cause of cancer-related deaths among women globally, highlighting the critical role of early detection through mammography screening. However, the effectiveness of mammography significantly depends on the accuracy of breast positioning. Incorrect positioning can lead to diagnostic errors, increased patient distress, and unnecessary additional imaging and costs.

Despite profound advancements in deep learning for breast cancer diagnostics, there has been a noticeable gap in tools specifically aimed at assessing the quality of mammogram positioning. Our paper addresses this gap by introducing a novel deep learning approach that quantitatively assesses the positioning quality of mediolateral oblique (MLO) mammograms. Utilizing advanced techniques such as attention mechanisms and coordinate convolution modules, our method identifies crucial anatomical landmarks like the nipple and pectoralis muscle, and automatically delineates the posterior nipple line (PNL).

This GitHub repository contains the source code, models, and instructions necessary for deploying and studying the task of mammography positioning.

## Citation
If you use this software, data, or methodology in your research, please cite as follows:
```
@article{tanyel2024mammographic,
  title={Mammographic Breast Positioning Assessment via Deep Learning},
  author={Tanyel, Toygar and Denizoglu, Nurper and Seker, Mustafa Ege and Alis, Deniz and Karaarslan, Ercan and Aribal, Erkin and Oksuz, Ilkay},
  journal={MICCAI, Deep-Brea3th 2024: A Deep Breast Workshop on AI and Imaging for Diagnostic and Treatment Challenges in Breast Care},
  volume={},
  number={},
  pages={},
  year={2024},
  publisher={}
}
```
Will be updated with publication.

## Labels
For detailed descriptions of the labels, visit [this link](https://github.com/tanyelai/deep-breast-positioning/tree/main/labels).


## Installation
To set up the project environment:
```bash
git clone https://github.com/tanyelai/deep-breast-positioning.git
cd deep-breast-positioning
```

## Performance

### Distance Errors in Millimeters (mm)

Distance errors are presented as mean (μ), standard deviation (σ), and median (x∼) to mitigate the influence of challenging cases (primarily due to subjectivity of the task).

| Models         | Perp μ | Perp σ | Perp x∼ | Pec1 μ | Pec1 σ | Pec1 x∼ | Pec2 μ | Pec2 σ | Pec2 x∼ | Nipple μ | Nipple σ | Nipple x∼ | Angular μ | Angular σ | Angular x∼ |
|----------------|--------|--------|---------|--------|--------|---------|--------|--------|---------|----------|----------|-----------|-----------|-----------|------------|
| R-ResNeXt50    | 7.13   | 4.23   | 6.49    | 7.33   | 6.01   | 5.24    | 7.93   | 7.00   | 6.20    | 4.63     | 1.99     | 4.45      | 2.71      | 2.44      | 1.96       |
| UNet           | 9.62   | 7.86   | 8.03    | 8.19   | 6.89   | 6.01    | 14.01  | 14.01  | 10.9    | 6.80     | 5.25     | 5.72      | 3.52      | 3.15      | 2.66       |
| Attention UNet | 5.12   | 5.04   | 3.56    | 6.01   | 5.87   | 4.03    | 6.94   | 8.25   | 3.95    | 2.98     | 2.40     | 2.52      | 2.58      | 2.73      | 1.81       |
| CoordAtt UNet  | 4.99   | 4.88   | 3.82    | 5.62   | 5.29   | 4.14    | 6.49   | 7.37   | 4.26    | 2.97     | 2.46     | 2.45      | 2.42      | 2.56      | 1.75       |

### Test Results on Automatically Generated Quality Labels

Test results on automatically generated quality labels extracted from radiologists' label drawings. The raw ResNeXt50 model was trained for binary classification based on image-level labels. The R-ResNeXt50 model had its last layer modified to function as a landmark regressor, predicting coordinates and overall positioning quality, similar to our proposed pipeline. Results are presented as the mean ± standard deviation of 5 different training runs.

| Model         | Accuracy            | Specificity         | Sensitivity         |
|---------------|---------------------|---------------------|---------------------|
| ResNeXt50     | 73.7 ± 3.35         | 76.91 ± 6.26        | 68.57 ± 11.41       |
| R-ResNeXt50   | 82.3 ± 5.03         | 81.42 ± 12.34       | 83.38 ± 10.49       |
| UNet          | 70.63 ± 1.49        | 78.46 ± 1.56        | 58.12 ± 2.68        |
| Attention UNet| 88.2 ± 2.51         | 88.62 ± 4.11        | **87.53 ± 3.51**    |
| CoordAtt UNet | **88.63 ± 2.84**    | **90.25 ± 4.04**    | 86.04 ± 3.41        |

## Example Predictions
<img width="983" alt="image" src="https://github.com/tanyelai/deep-breast-positioning/assets/44132720/2307adc8-95b1-4805-b8d4-5fb9e1107967">



## Contributing
Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.

Please ensure to update tests as appropriate.

## Contact
For questions or further inquiries about the code, please reach out at tanyel23@itu.edu.tr.
