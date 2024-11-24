# Weasyprint
import dataframe_image as dfi

dfi.export(view_res, os.path.join(img_dir, 'stat view results.png'))



from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os


# image_files = [os.path.join(img_dir, filename) for filename in os.listdir(image_folder) if filename.endswith(('.png'))]
image_files = [
'./output/2020-09-09_2023-09-09\\img\\Asset Prices.png',
'./output/2020-09-09_2023-09-09\\img\\Normalized Asset Prices.png',
'./output/2020-09-09_2023-09-09\\img\\Price Ratio.png',
'./output/2020-09-09_2023-09-09\\img\\Reg Resid Z Score.png',
'./output/2020-09-09_2023-09-09\\img\\Correlation.png',
'./output/2020-09-09_2023-09-09\\img\\subsample correlation.png',
'./output/2020-09-09_2023-09-09\\img\\stat view results.png',
]


from PIL import Image  # install by > python3 -m pip install --upgrade Pillow  # ref. https://pillow.readthedocs.io/en/latest/installation.html#basic-installation

images = [
    Image.open(f).convert('RGB')
    for f in image_files
]

pdf_path = os.path.join('output', runtime_dir, 'images.pdf')
    
images[0].save(
    pdf_path, "PDF" ,resolution=100.0, save_all=True, append_images=images[1:]
)



# print(f'Images in {image_folder} have been converted to {pdf_file}')
