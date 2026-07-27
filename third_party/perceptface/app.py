import gradio as gr
import cv2
from process_image import process_image
def to_black(image,cropped):
    print(f"Is the image cropped? {cropped}")
    output_image = process_image(image, cropped)
    return output_image

interface = gr.Interface(fn=to_black,   
                            inputs=[
                                        gr.Image(type="numpy", label="Original Photo"),  # 输入图像
                                        gr.Checkbox(label="If it is already a cropped face (dataset), check it", value=False)  # 是否裁剪的选项
                                    ],
                         
                         outputs=gr.Image(type="numpy",label="Protected Photo"))
interface.launch(share=True)
