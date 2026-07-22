import galsim


def get_galsim_image():
    pixel_scale = 0.2  # arcsec / pixel
    image_size = 128

    image = galsim.ImageF(image_size, image_size, scale=pixel_scale)
    psf = galsim.Airy(lam=532.0, diam=1.5, scale_unit=galsim.arcsec)

    grid_spacing = 15.0  # pixels
    flux_per_source = 2_500  # photons

    grid_size = 5
    offset_start = -(grid_size - 1) / 2.0 * grid_spacing

    for i in range(grid_size):
        for j in range(grid_size):
            x_offset = offset_start + i * grid_spacing
            y_offset = offset_start + j * grid_spacing

            source = galsim.DeltaFunction(flux=flux_per_source)
            csource = galsim.Convolve([source, psf])

            csource.drawImage(
                image=image, offset=(x_offset, y_offset), add_to_image=True
            )

    background_level = 10.0
    image += background_level

    image.addNoise(galsim.PoissonNoise())

    read_noise_sigma = 4.0
    image.addNoise(galsim.GaussianNoise(sigma=read_noise_sigma))

    return image.array
