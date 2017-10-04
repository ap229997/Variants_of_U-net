# Variants_of_U-net
Implementations of different variations of U-net - adding deconv layers, dense net variant and including skip connections for the task of **generating binary masks**. Also, used alpha matting for post-processing.

Train the model using the implementation files provided and while testing modify the architecture according to required variant in testing.py.

#### Obeservations:
> - The skip connection variant gave best result compared to other variants
> - Introducing skip connections led to ~3x improvement in speed 
