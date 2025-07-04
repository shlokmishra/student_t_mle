# Add these lines for debugging
echo "--- Environment Check ---"
which python
python -c "import numpy; print('NumPy Version:', numpy.__version__)"
echo "-------------------------"


echo "--- Running k=3, m=20 ---"
python3 main.py --k 3 --m 20
