import random
import galsim


def get_galsim_image(
    image_size=128, grid_dimensions=(8, 8), source_spacing_pixels=16, counts_per_source=2_500, psf_sigma=1.0, dropout_rate=0.25,
):
    # DEPRECATE
    pixel_scale = 0.2  # arcsec / pixel

    image = galsim.ImageF(image_size, image_size, scale=pixel_scale)
    psf = galsim.Gaussian(sigma=psf_sigma)

    # TODO ... 
    nx, ny = grid_dimensions
    
    start_x = -(nx - 1) / 2.0 * source_spacing_pixels
    start_y = -(ny - 1) / 2.0 * source_spacing_pixels

    source = galsim.DeltaFunction(flux=counts_per_source)
    csource = galsim.Convolve([source, psf])

    for i in range(nx):
        for j in range(ny):
            if random.random() < dropout_rate:
                continue

            x_offset = start_x + i * source_spacing_pixels
            y_offset = start_y + j * source_spacing_pixels

            csource.drawImage(
                image=image, offset=(x_offset, y_offset), add_to_image=True
            )

    background_level = 10.0
    image += background_level

    image.addNoise(galsim.PoissonNoise())

    read_noise_sigma = 4.0
    image.addNoise(galsim.GaussianNoise(sigma=read_noise_sigma))

    return image.array
