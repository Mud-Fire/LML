from itkwidgets import view
import itkwidgets
import nibabel as nib

nii = nib.load("./demo_image/1/sa.nii.gz")

# img = itk.imread(nii)

img = nii.get_fdata()
viewer = view(img, cmap=itkwidgets.cm.grayscale)

viewer