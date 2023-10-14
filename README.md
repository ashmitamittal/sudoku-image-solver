# Sudoku Image Solver
A project that effortlessly transforms sudoku puzzles from images into solved boards with the integration of OpenCV and advanced AI search techniques.

<br>
This project harnesses OpenCV's image processing capabilities to detect Sudoku boards within images. Once identified, the puzzles are tackled using a backtracking search algorithm which is further enhanced by forward checking and integrates three constraint satisfaction problem heuristics to optimize the ordering of variables and values:

- Most constrained variable
- Most constraining variable
- Least constraining value
  
These AI-driven techniques ensure efficient and accurate solutions for Sudoku puzzles.

## Overview
Here's a step-by-step rundown of what the Sudoku Image Solver does:

1. **Detection**: Identifies a Sudoku board within a provided image.
2. **Extraction**: Zooms into the board, focusing solely on the puzzle.
3. **Recognition**: Using a pre-trained OCR model specialized for Sudoku, it discerns each given number on the board.
4. **Solution**: Implements a backtracking search, forward-checking algorithms, and other heuristics to solve the puzzle.
5. **Overlay**: Post solution, it superimposes the solved numbers onto the original image, filling in the blanks.

All these operations are seamlessly integrated via OpenCV, Python, and deep-learning OCR detection.

## Running the Application

Run the following command to start the application:

```bash
python app.py
```

Additionally, I have extended this project to include a web interface built using Flask, allowing users to easily access and solve their Sudoku puzzles online.

## Sudoku Puzzle Solved Example
<img width="350" alt="image" src="https://github.com/ashmitamittal/sudoku-image-solver/assets/83453304/9250db88-5632-4936-b27f-e1607918c910">
<img width="350" alt="image" src="https://github.com/ashmitamittal/sudoku-image-solver/assets/83453304/72881be3-7e57-43ea-a7aa-e7ae43337123">


## Acknowledgements
Much of the image processing, especially the aspects related to reading the Sudoku grid and overlaying solutions, was inspired by this OpenCV [Sudoku Solver tutorial](https://data-flair.training/blogs/opencv-sudoku-solver/). It offered a deep dive into OpenCV's capabilities.
