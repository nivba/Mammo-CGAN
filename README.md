# Mammo-CGAN
Tumor injection tool for mammographic scans.

Project resentation - https://www.youtube.com/watch?v=bOTE8LqXQCg&t=27s (Hebrew)

#
My training data set consists 2880 images augmented from 36 images of tumors marked by a specialist radiologist

Tumor injection steps:
1. Crooping a square of an healthy tissue from a mammogram image.
2. Resizing the square to 64x64 pixels.
3. Conditioning the Image by zeroing the middel (44X44 pixel).
4. Feeding the condition image to the first GAN.
5. Feeding the first GAN output as input to the second GAN.
6. Resizing the output to its original size.
7. Adding a low power white gaussian noise for hiding the resize distortions.
8. Merging the fake image into the original image using a weighted average.

