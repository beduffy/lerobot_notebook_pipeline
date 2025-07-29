# Headless Testing Guide

## Issue Summary

When running tests in headless/cloud environments, 9 tests failed due to:

1. **TorchCodec/FFmpeg Dependencies**: `torchcodec` requires FFmpeg libraries that aren't available in headless environments
2. **Video Backend**: The default video backend (`torchcodec`) doesn't work without proper FFmpeg installation
3. **Display Dependencies**: Some visualization components try to access display

## Solution Implemented

### 1. Fixed Video Backend
- **Problem**: `torchcodec` requires FFmpeg libraries (`libavutil.so.*`, `libavdevice.so.*`)
- **Solution**: Force use of `pyav` backend in test fixtures via `conftest.py`

### 2. Headless Configuration
- **Problem**: Matplotlib and GUI components fail in headless environment
- **Solution**: Set `matplotlib.use('Agg')` and disable display in `conftest.py`

### 3. Updated Dependencies
- **Problem**: Missing headless-compatible video processing libraries
- **Solution**: Added `opencv-python-headless`, `imageio[ffmpeg]`, `pyav` to requirements

## Files Modified

1. **`tests/conftest.py`** (NEW):
   - Centralized test configuration
   - Forces `pyav` video backend
   - Sets up headless matplotlib
   - Provides fixtures with headless-compatible settings

2. **`requirements-cloud.txt`**:
   - Changed `opencv-python` → `opencv-python-headless`
   - Added `imageio[ffmpeg]` for video processing
   - Added `pyav` support

3. **Test files**:
   - Removed duplicate fixture definitions
   - Now use centralized fixtures from `conftest.py`

## How to Run Tests

### Standard Run (All Tests)
```bash
cd lerobot_notebook_pipeline
python -m pytest tests/ -v
```

### Fast Run (Headless-Specific)
```bash
cd lerobot_notebook_pipeline
python -m pytest tests/ -v -s --tb=short
```

### Single Test (For Debugging)
```bash
cd lerobot_notebook_pipeline
python -m pytest tests/test_analysis.py::test_get_dataset_stats -v -s
```

## Expected Results

**Before Fix**: 9 failed, 6 passed
- All failures due to `RuntimeError: Could not load libtorchcodec`

**After Fix**: 15 passed (or mostly passed)
- Video backend set to `pyav` instead of `torchcodec`
- Headless matplotlib backend configured
- Tests should run without FFmpeg dependency issues

## Fallback Options

If tests still fail, try:

1. **Skip Video Tests**:
   ```bash
   python -m pytest tests/ -v -k "not visualization and not sample"
   ```

2. **Environment Variables**:
   ```bash
   export MPLBACKEND=Agg
   export DISPLAY=""
   python -m pytest tests/ -v
   ```

3. **Manual Test**:
   ```bash
   python test_headless.py
   ```

## Key Learnings

1. **Cloud environments** often lack FFmpeg libraries needed by TorchCodec
2. **PyAV backend** is more reliable for headless operations
3. **Centralized test configuration** via `conftest.py` ensures consistency
4. **Headless matplotlib** prevents GUI-related failures

The fixes ensure that the test suite works reliably in cloud/headless environments like Lightning AI, Google Colab, AWS, etc. 