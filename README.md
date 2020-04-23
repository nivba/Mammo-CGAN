# Mammo-CGAN
Tumor injection tool for mammographic scans.

This project is still on progress, First working example: https://www.youtube.com/watch?v=5sGZoZa4FY4

#
My training data set consists 2880 images augmented from 36 images of tumors marked by a specialist radiologist

Tumor injection steps:
1. Crooping a square of a healthy tissue from a mammogram image.
2. Resizing the square to be 64x64 pixels.
3. Conditioning the Image by zeroing the middel (44X44 pixel).
4. Feeding the condition image to the first GAN.
5. Feeding the first GAN output as input to the second GAN.
6. Resizing the output to its original size.
7. Adding a low power white gaussian noise for hiding the resize distortions.
8. Merging the fake image into the original image using a weighted average.

