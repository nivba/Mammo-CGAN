# Mammo-CGAN
Tumor injection tool for mammographic scans.

This project is still on progress. first results are available in the repository Wiki.

First working example: https://www.youtube.com/watch?v=4OOV6HlChI0

#
My training data set consists 2880 images augmented from 36 images of tumors marked by a specialist radiologist

Tumor injection steps:
1. Crooping a square of a healthy tissue from a mammogram image.
2. Resize the square to be 64x64 pixels.
3. Condition the Image- zeroing the middel of the image (44X44 pixel).
4. Feeding the condition image to the first GAN.
5. Feeding the first GAN output as input to the second GAN.
6. Resize the output to its original size.
7. Adding a low power white gaussian noise for hiding the resize artifacts.
8. Merging the fake image into the original image using a weighted average.

