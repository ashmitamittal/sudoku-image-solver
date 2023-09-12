from flask import Flask, render_template, request, redirect, url_for, flash
import os
import cv2
import numpy as np
from solve_image_sudoku import *

app = Flask(__name__)
app.secret_key = 'some_secret_key'  # for flash messages

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER


@app.route('/', methods=['GET', 'POST'])
def index():
    # Clear the folders when the user accesses the home page
    clear_folders()
    
    if request.method == 'POST':
        # Check for file in the request
        file = request.files.get('file')
        if not file or file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        try:
            # Processing the image and saving the result
            result_image = solve_image_sudoku(filename)
            
            if result_image is None:
                raise ValueError('Error processing the image.')
            
            result_image_path = os.path.join(app.config['RESULT_FOLDER'], "result_" + file.filename)
            os.remove(filename)  # Delete the uploaded image
            redirect_url = url_for('result', image_name="result_" + file.filename)
            print("Redirecting to:", redirect_url)
            return redirect(redirect_url)

        except Exception as e:
            print(f"Error: {e}")
            flash('There was an error processing the image. Please ensure it is a Sudoku image.')
            return redirect(url_for('index'))

    return render_template('index.html')


@app.route('/result/<image_name>')
def result(image_name):
    return render_template('result.html', image=os.path.join("results", image_name).replace("\\", "/"))


@app.route('/delete/<image_name>')
def delete_and_redirect(image_name):
    print("nside detetel")
    image_path = os.path.join(app.config['RESULT_FOLDER'], image_name)
    print("image path to be deleted: ", image_path)
    if os.path.exists(image_path):
        print("deleted, image", image_path)
        os.remove(image_path)
    
    return redirect(url_for('index'))


def clear_folders():
    folders = [app.config['UPLOAD_FOLDER'], app.config['RESULT_FOLDER']]
    for folder in folders:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}. Reason: {e}")


def solve_image_sudoku(image_path):
    # Read image
    img = cv2.imread(image_path)

    # Extract board from input image
    board, location = find_board(img)

    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    rois = split_boxes(gray)
    rois = np.array(rois).reshape(-1, input_size, input_size, 1)

    # Get prediction
    prediction = model.predict(rois)
    predicted_numbers = [classes[np.argmax(i)] for i in prediction]

    # Reshape the list 
    board_num = np.array(predicted_numbers).astype('uint8').reshape(9, 9)

    result_filename = os.path.join(app.config['RESULT_FOLDER'], "result_" + os.path.basename(image_path))

    # Solve the board
    try:
        solved_board_nums = get_board(board_num)
        binArr = np.where(np.array(predicted_numbers) > 0, 0, 1)
        flat_solved_board_nums = solved_board_nums.flatten() * binArr
        mask = np.zeros_like(board)
        solved_board_mask = displayNumbers(mask, flat_solved_board_nums)
        inv = get_InvPerspective(img, solved_board_mask, location)
        combined = cv2.addWeighted(img, 0.7, inv, 1, 0)
        cv2.imwrite(result_filename, combined)  # Save the processed image

    except Exception as e:
        print("Error:", e)
        flash("Solution doesn't exist. Model misread digits.")
        return None  # Returning None when there's an error

    return result_filename  # Return the path of the saved result image



if __name__ == '__main__':
    app.run(debug=True)
