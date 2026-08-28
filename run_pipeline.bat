@echo off
REM ============================================================================
REM RARE26-Oak Full Training Pipeline
REM ============================================================================
REM
REM This script runs the complete 3-stage training pipeline:
REM   Stage 1: Train mother models (ConvNext-Tiny, ConvNext-Base, EfficientNetV2)
REM   Stage 2: Generate pseudo labels from mother model ensemble
REM   Stage 3: Train champion models (DINOv3-ConvNeXt, ResNet50, ViT-Base)
REM
REM USAGE:
REM   run_pipeline.bat              (full training - requires GPU + real data)
REM   run_pipeline.bat --debug      (debug mode: 1 epoch, 16 samples, tests pipeline flow)
REM
REM ============================================================================

REM ============================================================================
REM USER CONFIGURATION - Edit these values to match your setup
REM ============================================================================

REM -----------------------------------------------------------------------
REM DEBUG FLAG: Set to "true" to test the pipeline without real training.
REM   true  = 1 epoch, 16 samples only. Use this to verify the pipeline works.
REM   false = Full training with all epochs and all data.
REM -----------------------------------------------------------------------
SET DEBUG=false

REM Override with command line --debug flag if provided
IF "%1"=="--debug" SET DEBUG=true

REM Data directories (relative to project root)
SET RARE26_DATA_DIR=data\rare26
SET GASTRONET_DATA_DIR=data\gastronet

REM Pre-trained weight paths (uncomment and set these if you have them)
REM SET GASTRONET_WEIGHTS=base_model\RN50_Billion-Scale-SWSL2BGastroNet-5M_DINOv1.pth
REM SET DINOV2_WEIGHTS=base_model\dinov2.pth

REM Output directories
SET MOTHER_WEIGHTS_DIR=outputs\mother_weights
SET PSEUDO_LABELS_DIR=outputs\pseudo_labels
SET CHAMP_WEIGHTS_DIR=outputs\champ_weights

REM ============================================================================
REM Setup
REM ============================================================================
SET DEBUG_FLAG=
IF "%DEBUG%"=="true" (
    SET DEBUG_FLAG=--debug
    echo ============================================================================
    echo  [DEBUG MODE] Running with 1 epoch and 16 samples for testing.
    echo  Set DEBUG=false in this script for real training.
    echo ============================================================================
    echo.
)

REM Add src to PYTHONPATH so the package is importable
SET PYTHONPATH=src;%PYTHONPATH%

REM Ensure output directories exist
IF NOT EXIST "%MOTHER_WEIGHTS_DIR%" mkdir "%MOTHER_WEIGHTS_DIR%"
IF NOT EXIST "%PSEUDO_LABELS_DIR%" mkdir "%PSEUDO_LABELS_DIR%"
IF NOT EXIST "%CHAMP_WEIGHTS_DIR%" mkdir "%CHAMP_WEIGHTS_DIR%"

echo ============================================================================
echo  RARE26-Oak Training Pipeline
echo  DEBUG=%DEBUG%
echo  Device: will auto-detect (CUDA if available, else CPU)
echo ============================================================================
echo.

REM ============================================================================
REM STAGE 1: Train Mother Models
REM ============================================================================
echo [STAGE 1/3] Training mother models (ConvNext-Tiny, ConvNext-Base, EfficientNetV2)...
echo.

python -m rare26_oak.cli.train_mother ^
    --model all ^
    --data-dir %RARE26_DATA_DIR% ^
    --output-dir %MOTHER_WEIGHTS_DIR% ^
    %DEBUG_FLAG%

IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Stage 1 failed! Aborting pipeline.
    exit /b 1
)
echo.
echo [STAGE 1/3] Mother model training complete.
echo.

REM ============================================================================
REM STAGE 2: Generate Pseudo Labels
REM ============================================================================
echo [STAGE 2/3] Generating pseudo labels from mother model ensemble...
echo.

python -m rare26_oak.cli.generate_pseudo_labels ^
    --gastronet-dir %GASTRONET_DATA_DIR% ^
    --mother-weights-dir %MOTHER_WEIGHTS_DIR% ^
    --output-dir %PSEUDO_LABELS_DIR% ^
    %DEBUG_FLAG%

IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Stage 2 failed! Aborting pipeline.
    exit /b 1
)
echo.
echo [STAGE 2/3] Pseudo label generation complete.
echo.

REM ============================================================================
REM STAGE 3: Train Champion Models
REM ============================================================================
echo [STAGE 3/3] Training champion models (ConvNeXt, ResNet50, ViT-Base)...
echo.

SET EXTRA_ARGS=
IF DEFINED GASTRONET_WEIGHTS SET EXTRA_ARGS=%EXTRA_ARGS% --gastronet-weights %GASTRONET_WEIGHTS%
IF DEFINED DINOV2_WEIGHTS SET EXTRA_ARGS=%EXTRA_ARGS% --dinov2-weights %DINOV2_WEIGHTS%

python -m rare26_oak.cli.train_champion ^
    --model all ^
    --data-dir %RARE26_DATA_DIR% ^
    --pseudo-csv %PSEUDO_LABELS_DIR%\pseudo_labels.csv ^
    --output-dir %CHAMP_WEIGHTS_DIR% ^
    %EXTRA_ARGS% ^
    %DEBUG_FLAG%

IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Stage 3 failed! Aborting pipeline.
    exit /b 1
)
echo.
echo [STAGE 3/3] Champion model training complete.
echo.

echo ============================================================================
echo  PIPELINE COMPLETE!
echo.
echo  Mother weights:     %MOTHER_WEIGHTS_DIR%
echo  Pseudo labels:      %PSEUDO_LABELS_DIR%\pseudo_labels.csv
echo  Champion weights:   %CHAMP_WEIGHTS_DIR%
echo ============================================================================
