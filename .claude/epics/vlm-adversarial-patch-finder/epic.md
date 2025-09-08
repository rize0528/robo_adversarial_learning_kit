---
name: vlm-adversarial-patch-finder
status: backlog
created: 2025-09-08T08:37:21Z
progress: 0%
prd: .claude/prds/vlm-adversarial-patch-finder.md
github: https://github.com/rize0528/robo_adversarial_learning_kit/issues/1
---

# Epic: VLM Adversarial Patch Finder

## Overview
Build an AI robot model security demonstration system for exhibitions that showcases how simple adversarial patches can compromise vision-language models in robotics applications. The system generates optimized adversarial patches and demonstrates their real-time attack effects on VLMs, educating stakeholders about critical AI security vulnerabilities.

## Architecture Decisions

### Core Technology Stack
- **VLM Framework**: HuggingFace Transformers with Gemma-3 4B/12B models for balance between capability and resource requirements
- **Optimization Engine**: PyTorch for gradient-based adversarial patch generation with white-box access
- **Inference Runtime**: Ollama or LiteLLM for local model deployment without cloud dependencies
- **Camera Interface**: OpenCV for cross-platform camera integration and frame processing

### Design Patterns
- **Modular Pipeline**: Separate patch generation and inference modules for independent development and testing
- **Plugin Architecture**: Configurable loss functions and attack strategies through abstract interfaces
- **State Management**: Simple file-based persistence for patches and configuration (no database needed)
- **Exhibition Mode**: Streamlined UI with preset scenarios for non-technical booth operators

### Simplification Strategies
- **Leverage Existing Tools**: Use HuggingFace pipelines instead of custom model loading code
- **Pre-computed Patches**: Generate patches offline, load them for demo (reduces live computation)
- **Static Scenarios**: Pre-define 3 exhibition scenarios instead of dynamic configuration
- **Simple UI**: Terminal-based or minimal web interface using Gradio/Streamlit

## Technical Approach

### Frontend Components
- **Demo Interface**: Gradio web app with side-by-side normal/attacked output display
- **Camera View**: Live webcam feed with overlay showing patch detection status
- **Attack Selector**: Simple dropdown for healthcare/manufacturing/autonomous scenarios
- **Impact Dashboard**: Pre-written security impact explanations for each scenario

### Backend Services
- **VLM Service**: Single inference endpoint handling both normal and patched images
- **Patch Generator**: Offline script for adversarial patch optimization
- **Camera Handler**: Frame capture and preprocessing pipeline
- **Attack Detector**: Simple template matching to identify when patch is visible

### Infrastructure
- **Local Deployment**: Single machine setup with no external dependencies
- **Resource Management**: Model lazy loading to optimize memory usage
- **Logging**: Simple file-based logging for debugging and metrics
- **Configuration**: YAML files for easy parameter tuning

## Implementation Strategy

### Development Phases
1. **Foundation (Week 1-2)**: Environment setup, model loading, basic inference
2. **Patch Generation (Week 3-4)**: Optimization algorithm, loss functions, patch creation
3. **Demo Pipeline (Week 5-6)**: Camera integration, real-time inference, UI development
4. **Exhibition Polish (Week 7)**: Scenario presets, reliability testing, documentation

### Risk Mitigation
- **Model Fallback**: Support both 4B and 12B models, default to smaller if memory constrained
- **Offline Generation**: Pre-generate patches to avoid live optimization failures
- **Graceful Degradation**: System continues with static images if camera fails
- **Simple Recovery**: One-button restart for booth operators

### Testing Approach
- **Unit Tests**: Core functions (model loading, loss calculation, image processing)
- **Integration Tests**: End-to-end patch generation and attack demonstration
- **Exhibition Simulation**: 8-hour continuous run tests for reliability
- **User Testing**: Non-technical operators validate setup and operation

## Task Breakdown Preview

Simplified to essential tasks (maximum 10):

- [ ] **T1: Environment & Model Setup** - Install dependencies, load Gemma-3 VLM, verify inference
- [ ] **T2: Data Pipeline** - Load training images, implement preprocessing, create data loaders
- [ ] **T3: Patch Optimization Core** - Implement gradient-based optimization with PyTorch
- [ ] **T4: Loss Functions** - Create targeted and non-targeted attack loss functions
- [ ] **T5: Patch Generation Script** - Complete offline patch generation workflow
- [ ] **T6: Camera Integration** - Capture frames, detect patches, process for VLM
- [ ] **T7: Demo Interface** - Build Gradio app with attack scenarios and visualizations
- [ ] **T8: Exhibition Presets** - Configure 3 scenarios (healthcare/manufacturing/autonomous)
- [ ] **T9: Testing & Validation** - Test patch effectiveness, system reliability, performance
- [ ] **T10: Documentation & Packaging** - Setup guide, operator manual, deployment package

## Dependencies

### External Services
- **HuggingFace Hub**: Model weights download (one-time, can be cached)
- **Python Package Index**: Standard ML/CV libraries installation

### Internal Prerequisites
- **Hardware**: 16GB+ RAM machine with webcam
- **Software**: Python 3.8+, CUDA optional for acceleration
- **Data**: 10-50 annotated training images for patch generation

### Critical Path Items
- Model availability from HuggingFace (Week 1)
- Training data collection and annotation (Week 2)
- Physical patch printing for testing (Week 6)

## Success Criteria (Technical)

### Performance Benchmarks
- **Inference Speed**: <5 seconds per frame on CPU
- **Attack Success**: >70% effectiveness on test images
- **System Reliability**: 8-hour continuous operation without crashes
- **Memory Usage**: <16GB RAM peak consumption

### Quality Gates
- All unit tests passing (Week 4)
- Successful patch generation on 3+ scenarios (Week 5)
- Exhibition simulation complete (Week 7)
- Operator can run full demo independently (Week 7)

### Acceptance Criteria
- Clear visual demonstration of AI model compromise
- Non-technical stakeholders understand security implications
- System runs reliably in exhibition environment
- Setup time <30 minutes for booth deployment

## Estimated Effort

### Overall Timeline
- **Total Duration**: 7-8 weeks from start to exhibition-ready
- **Development Effort**: 2 developers working in parallel
- **Critical Milestones**: 
  - Week 4: First successful patch generation
  - Week 6: Complete demo pipeline
  - Week 7: Exhibition ready

### Resource Requirements
- **ML Engineer**: 60% allocation for optimization and model work
- **Software Developer**: 40% allocation for UI and integration
- **Testing Support**: 1 week for validation and documentation

### Risk Buffer
- 1 week contingency for model compatibility issues
- 3 days for physical patch printing and testing
- 2 days for exhibition setup and rehearsal

## Tasks Created
- [ ] #3 - Environment & Model Setup (parallel: false)
- [ ] #10 - Data Pipeline (parallel: false, depends on #3)
- [ ] #11 - Patch Optimization Core (parallel: false, depends on #3, #10)
- [ ] #5 - Loss Functions (parallel: true, can run with #11)
- [ ] #7 - Patch Generation Script (parallel: false, depends on #11, #5)
- [ ] #8 - Camera Integration (parallel: true, can run independently)
- [ ] #2 - Demo Interface (parallel: false, depends on #3, #8)
- [ ] #4 - Exhibition Presets (parallel: false, depends on #7, #2)
- [ ] #6 - Testing & Validation (parallel: false, depends on all)
- [ ] #9 - Documentation & Packaging (parallel: false, depends on #6)

Total tasks: 10
Parallel tasks: 2 (issues #5, #8)
Sequential tasks: 8
Estimated total effort: 86 hours
