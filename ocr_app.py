from requests.models import MissingSchema
import streamlit as st
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO
import base64
import easyocr
import pandas as pd

image_read = np.zeros([100, 100, 3], dtype=np.uint8)
framework = None
language = None
output_image = None
output_text_processed = None


@st.cache
def load_model():

    print("Framework Selected: ", framework)
    print("Languages Selected: ", language)
    model = None

    if framework == "EasyOCR":

        if 'English' or 'French' or 'Dutch' or 'German' in language:
            model = easyocr.Reader(lang_list=['en', 'fr', 'nl', 'de'], model_storage_directory='frameworks/easyocr/model',
                                   download_enabled=False, gpu=False)
        else:
            print("Easyocr model not loaded. Check the language support!!!")

    else:
        print("Please check the coding for the framework requested")

    return model


def process_text_with_score(results):

    individual_result = []

    if framework == "EasyOCR":

        for each_result in results:
            text_recognized = each_result[1]
            confidence_score = round(each_result[2] * 100, 2)
            text_confidence = (text_recognized, str(confidence_score) + '%')
            individual_result.append(text_confidence)

        print(individual_result)

    return individual_result


def display_output_image_with_bbox_text(results):

    output_canvas = np.full(image_read.shape[:3], 255, dtype=np.uint8)
    combined_result = output_canvas

    if framework == 'EasyOCR':

        bbox = [i[0] for i in results]
        text_recognized = [j[1] for j in results]

        for box, text in zip(bbox, text_recognized):

            pts = np.array(box, np.int32)
            pts = pts.reshape((-1, 1, 2))

            rec_result = text
            # print(recResult)

            # Get scaled values.
            box_height = int((abs((box[0][1] - box[3][1]))))

            # Get scale of the font.
            font_scale = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SIMPLEX, box_height-10, 1)

            # Write the recognized text on the output image.
            placement = (int(box[0][0]), int(box[0][1]))

            cv2.putText(output_canvas, rec_result, placement,
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 1, 5)

        # Draw the bounding boxes of text detected.
            cv2.polylines(image_read, [pts], True, (255, 0, 255), 4)

        # Concatenate the input image with the output image.
        combined_result = cv2.hconcat([image_read, combined_result])

    return combined_result


def get_image_download_link(img):
    img = Image.fromarray(np.uint8(img)).convert('RGB')
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="output.jpeg" target="_blank" >Download output image</a>'
    return href


def get_table_download_link_csv(df):
    csv = df.to_csv().encode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="output.csv" target="_blank">Download csv file</a>'
    return href


def result_framework():

    results = None

    if framework == "EasyOCR":
        st.spinner()
        with st.spinner('Extracting Text from given Image'):
            reader = load_model()
            results = reader.readtext(image_read, detail=1)

    else:
        print("Framework not supported")

    return results


def click_extract_text_button():

    global output_image
    global output_text_processed

    output_results = result_framework()
    output_image = display_output_image_with_bbox_text(output_results)
    output_text_processed = process_text_with_score(output_results)

    return


def display_output_in_app():

    st.subheader('Output')
    st.success('Success')
    st.image(output_image)
    df = pd.DataFrame(output_text_processed, columns=['Text Extracted', 'Confidence Score'])
    st.table(df)
    st.markdown(get_image_download_link(output_image), unsafe_allow_html=True)
    st.markdown(get_table_download_link_csv(df), unsafe_allow_html=True)

    return


def load_application():

    global image_read
    global framework
    global language

    st.sidebar.image("app_files/app.jpeg")

    st.markdown("<h1 style='text-align: center; color: red;'>Text Extraction Application</h1>",
                unsafe_allow_html=True)

    img_file_buffer = st.file_uploader("Upload an Image with Text", type=['jpg', 'jpeg', 'png'])
    st.markdown("<h6 style='text-align: center';>OR</h6>", unsafe_allow_html=True)
    url = st.text_input('Enter Image URL Details')

    framework = st.sidebar.radio('Select a framework', ['EasyOCR', 'Others'])
    language = st.sidebar.multiselect('Select language', ['English', 'French', 'German', 'Dutch'])

    if language and "Others" not in framework and img_file_buffer is not None:
        # Read the file and convert it to opencv Image.
        image_read = np.array(Image.open(img_file_buffer))
        st.image(image_read)

        if st.button("Extract Text"):
            click_extract_text_button()
            display_output_in_app()
            st.balloons()

        else:
            st.info("Click Extract Text button for result")

    elif language and "Others" not in framework and url != '':

        try:
            response = requests.get(url)
            image_read = np.array(Image.open(BytesIO(response.content)))
            st.image(image_read)

            if st.button("Extract Text"):
                click_extract_text_button()
                display_output_in_app()
                st.balloons()

            else:
                st.info("Click Extract text button for result")

        except MissingSchema as err:
            st.header('Invalid URL, Try Again!')
            print(err)

        except UnidentifiedImageError as err:
            st.header('URL has no Image, Try Again!')
            print(err)

    else:
        st.warning("1. Select Framework \n 2. Select Language \n 3. Upload an Image or Provide Image URL")


if __name__ == "__main__":

    load_application()
