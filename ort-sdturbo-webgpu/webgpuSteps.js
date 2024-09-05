"use strict"

const sigma = 14.6146;
const gamma = 0;
const vae_scaling_factor = 0.18215;

class SDTurboWebGPUSteps {
    device;
    canvas;
    ort;

    queueWebgpu;
    renderContext;
    textEncoderOutputsBuffer;
    latentBuffer;
    unetSampleInputsBuffer;
    unetOutSampleBuffer;
    decodedOutputsBuffer;
    prescaleLatentSpacePipeline;
    prescaleLatentSpaceBindGroup;
    stepLatentSpacePipeline;
    stepLatentSpaceBindGroup;
    renderPipeline;
    renderBindGroup;

    managedTensors;

    pixelHeight;
    pixelWidth;
    latent_shape;

    constructor(latent_shape = [1, 4, 64, 64], options = { pixelHeight: 512, pixelWidth: 512 }) {
        this.latent_shape = latent_shape;

        const { pixelHeight, pixelWidth } = options;

        this.pixelHeight = pixelHeight || 512;
        this.pixelWidth = pixelWidth || 512;

        this.buildShaderCode();
    }

    buildShaderCode() {
        this.PRESCALE_LATENT_SPACE_SHADER = `
        @binding(0) @group(0) var<storage, read_write> result: array<vec4<f32>>;
        @binding(1) @group(0) var<storage, read> latentData: array<vec4<f32>>;
        
        @compute @workgroup_size(128, 1, 1)
        fn _start(@builtin(global_invocation_id) GlobalId : vec3<u32>) {
          let index = GlobalId.x;
          let value = latentData[index] / 14.64877241136608;
          result[index] = value;
        }
        `;

        this.STEP_LATENT_SPACE_SHADER = `
        @binding(0) @group(0) var<storage, read_write> result: array<vec4<f32>>;
        @binding(1) @group(0) var<storage, read> latentData: array<vec4<f32>>;
        
        @compute @workgroup_size(128, 1, 1)
        fn _start(@builtin(global_invocation_id) GlobalId : vec3<u32>) {
          let index = GlobalId.x;
          let sigma_hat = ${sigma};
          let latentVal = latentData[index];
          let outputSampleVal = result[index];
        
          let pred_original_sample = latentVal - ${sigma} * outputSampleVal;
          let derivative = (latentVal - pred_original_sample) / ${sigma};
          let dt = -${sigma};
          result[index] = (latentVal + derivative * dt) / ${vae_scaling_factor};
        }
        `;

        this.VERTEX_SHADER = `
        struct VertexOutput {
          @builtin(position) Position : vec4<f32>,
          @location(0) fragUV : vec2<f32>,
        }
        
        @vertex
        fn main(@builtin(vertex_index) VertexIndex : u32) -> VertexOutput {
          var pos = array<vec2<f32>, 6>(
            vec2<f32>( 1.0,  1.0),
            vec2<f32>( 1.0, -1.0),
            vec2<f32>(-1.0, -1.0),
            vec2<f32>( 1.0,  1.0),
            vec2<f32>(-1.0, -1.0),
            vec2<f32>(-1.0,  1.0)
          );
        
          var uv = array<vec2<f32>, 6>(
            vec2<f32>(1.0, 0.0),
            vec2<f32>(1.0, 1.0),
            vec2<f32>(0.0, 1.0),
            vec2<f32>(1.0, 0.0),
            vec2<f32>(0.0, 1.0),
            vec2<f32>(0.0, 0.0)
          );
        
          var output : VertexOutput;
          output.Position = vec4<f32>(pos[VertexIndex], 0.0, 1.0);
          output.fragUV = uv[VertexIndex];
          return output;
        }
        `;

        this.PIXEL_SHADER = `
        @group(0) @binding(1) var<storage, read> buf : array<f32>;
        
        @fragment
        fn main(@location(0) fragUV : vec2<f32>) -> @location(0) vec4<f32> {
          // The user-facing camera is mirrored, flip horizontally.
          var coord = vec2(0.0, 0.0);
          if (fragUV.x < 0.5) {
              coord = vec2(fragUV.x + 0.5, fragUV.y);
          } else {
              coord = vec2(fragUV.x - 0.5, fragUV.y);
         }
        
          // NHWC
          let redInputOffset = 0;
          let greenInputOffset = ${this.pixelWidth * this.pixelHeight};
          let blueInputOffset = ${2 * this.pixelWidth * this.pixelHeight};
          // Note that coord is normalized to 0.0~1.0
          let index = i32(coord.x * f32(${this.pixelWidth})) + i32(coord.y * f32(${this.pixelWidth * this.pixelHeight}));
          let r = clamp(buf[index] / 2 + 0.5, 0.0, 1.0);
          let g = clamp(buf[greenInputOffset + index] / 2 + 0.5, 0.0, 1.0);
          let b = clamp(buf[blueInputOffset + index] / 2 + 0.5, 0.0, 1.0);
          let a = 1.0;
        
          var out_color = vec4<f32>(r, g, b, a);
        
          return out_color;
        }
        `
    }

    uploadToGPU(buffer, values, type) {
        const stagingBuffer = this.device.createBuffer({
            usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
            size: values.buffer.byteLength,
            mappedAtCreation: true
        });
        const arrayBuffer = stagingBuffer.getMappedRange();
        if (type === 'float32') {
            new Float32Array(arrayBuffer).set(values);
        } else if (type === 'int32') {
            new Int32Array(arrayBuffer).set(values);
        }
        stagingBuffer.unmap();
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(stagingBuffer, 0, buffer, 0, values.byteLength);
        this.device.queue.submit([encoder.finish()]);
        stagingBuffer.destroy();
    };

    submitLatentSpaceComputeTask(pipeline, bindGroup) {
        const latentShapeCumMul = this.latent_shape.reduce((a, b) => a * b, 1);
        const workgroupsForLatentSpace = latentShapeCumMul / 4 / 128;

        let commandEncoderWebgpu = this.device.createCommandEncoder();
        let computePassEncoder = commandEncoderWebgpu.beginComputePass();
        computePassEncoder.setPipeline(pipeline);
        computePassEncoder.setBindGroup(0, bindGroup);
        computePassEncoder.dispatchWorkgroups(workgroupsForLatentSpace, 1, 1);
        computePassEncoder.end();
        computePassEncoder = null;
        this.queueWebgpu.submit([commandEncoderWebgpu.finish()]);
    };

    async downloadToCPU(buffer) {
        const stagingBuffer = this.device.createBuffer({
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            size: buffer.size
        });
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, buffer.size);
        this.device.queue.submit([encoder.finish()]);

        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const arrayBuffer = stagingBuffer.getMappedRange().slice(0, buffer.size / 4);
        stagingBuffer.destroy();
        return new Float32Array(arrayBuffer);
    };

    webgpuResourceInitialize(device, canvas, ort) {
        this.device = device;
        this.canvas = canvas;
        this.ort = ort;

        this.queueWebgpu = this.device.queue;

        this.textEncoderOutputsBuffer = this.device.createBuffer({
            size: Math.ceil((1 * 77 * 1024 * 4) / 16) * 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });

        const latentShapeCumMul = this.latent_shape.reduce((a, b) => a * b, 1);

        this.unetOutSampleBuffer = this.device.createBuffer({
            size: Math.ceil((latentShapeCumMul * 4) / 16) * 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });

        this.latentBuffer = this.device.createBuffer({
            size: Math.ceil((latentShapeCumMul * 4) / 16) * 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });

        this.unetSampleInputsBuffer = this.device.createBuffer({
            size: Math.ceil((latentShapeCumMul * 4) / 16) * 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });

        this.decodedOutputsBuffer = this.device.createBuffer({
            size: Math.ceil((1 * 3 * this.pixelHeight * this.pixelWidth * 4) / 16) * 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });

        this.prescaleLatentSpacePipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: this.device.createShaderModule({
                    code: this.PRESCALE_LATENT_SPACE_SHADER,
                }),
                entryPoint: '_start',
            },
        });

        this.prescaleLatentSpaceBindGroup = this.device.createBindGroup({
            layout: this.prescaleLatentSpacePipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.unetSampleInputsBuffer,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.latentBuffer,
                    },
                }
            ],
        });

        this.stepLatentSpacePipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: this.device.createShaderModule({
                    code: this.STEP_LATENT_SPACE_SHADER,
                }),
                entryPoint: '_start',
            },
        });
        this.stepLatentSpaceBindGroup = this.device.createBindGroup({
            layout: this.stepLatentSpacePipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.unetOutSampleBuffer,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.latentBuffer,
                    },
                }
            ],
        });

        this.canvas.width = this.pixelWidth;
        this.canvas.height = this.pixelHeight;
        this.renderContext = this.canvas.getContext('webgpu');
        const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
        const presentationSize = [this.pixelWidth, this.pixelHeight];
        this.renderContext.configure({
            device: this.device,
            size: presentationSize,
            format: presentationFormat,
            alphaMode: 'opaque',
        });
        this.renderPipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: this.device.createShaderModule({
                    code: this.VERTEX_SHADER,
                }),
                entryPoint: 'main',
            },
            fragment: {
                module: this.device.createShaderModule({
                    code: this.PIXEL_SHADER,
                }),
                entryPoint: 'main',
                targets: [
                    {
                        format: presentationFormat,
                    },
                ],
            },
            primitive: {
                topology: 'triangle-list',
            },
        });

        this.renderBindGroup = this.device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 1,
                    resource: {
                        buffer: this.decodedOutputsBuffer,
                    },
                }
            ],
        });

        const textEncoderOutputsBuffer = this.textEncoderOutputsBuffer;
        const unetOutSampleBuffer = this.unetOutSampleBuffer;
        const unetSampleInputsBuffer = this.unetSampleInputsBuffer;
        const decodedOutputsBuffer = this.decodedOutputsBuffer;
        const textEncoderOutputsTensor = this.ort.Tensor.fromGpuBuffer(textEncoderOutputsBuffer, {
            dataType: 'float32', dims: [1, 77, 1024],
            dispose: () => textEncoderOutputsBuffer.destroy()
        });
        const unetOutSampleTensor = this.ort.Tensor.fromGpuBuffer(unetOutSampleBuffer, {
            dataType: 'float32', dims: this.latent_shape,
            dispose: () => unetOutSampleBuffer.destroy()
        });
        const unetSampleInputsTensor = this.ort.Tensor.fromGpuBuffer(unetSampleInputsBuffer, {
            dataType: 'float32', dims: this.latent_shape,
            dispose: () => unetSampleInputsBuffer.destroy()
        });
        const decodedOutputsTensor = this.ort.Tensor.fromGpuBuffer(decodedOutputsBuffer, {
            dataType: 'float32', dims: [1, 3, this.pixelHeight, this.pixelWidth],
            dispose: () => decodedOutputsBuffer.destroy()
        });

        this.managedTensors = { textEncoderOutputsTensor, unetOutSampleTensor, unetSampleInputsTensor, decodedOutputsTensor };
    };

    getManagedTensors() {
        return this.managedTensors;
    }

    setupRandomLatent() {
        const randn_latents = (shape, noise_sigma) => {
            const randn = () => {
                // Use the Box-Muller transform
                let u = Math.random();
                let v = Math.random();
                let z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
                return z;
            }
            let size = 1;
            shape.forEach(element => {
                size *= element;
            });

            let data = new Float32Array(size);
            // Loop over the shape dimensions
            for (let i = 0; i < size; i++) {
                data[i] = randn() * noise_sigma;
            }
            return data;
        }
        const latentData = randn_latents(this.latent_shape, sigma);
        this.uploadToGPU(this.latentBuffer, latentData, "float32");
        this.submitLatentSpaceComputeTask(this.prescaleLatentSpacePipeline, this.prescaleLatentSpaceBindGroup);
    }

    stepLatentSpace() {
        this.submitLatentSpaceComputeTask(this.stepLatentSpacePipeline, this.stepLatentSpaceBindGroup);
    };

    async stepDraw() {
        const commandEncoder = this.device.createCommandEncoder();
        const textureView = this.renderContext.getCurrentTexture().createView();
        const renderPassDescriptor = {
            colorAttachments: [
                {
                    view: textureView,
                    clearValue: { r: 1.0, g: 0.0, b: 0.0, a: 1.0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                },
            ],
        };

        const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
        passEncoder.setPipeline(this.renderPipeline);
        passEncoder.setBindGroup(0, this.renderBindGroup);
        passEncoder.draw(6, 1, 0, 0);
        passEncoder.end();
        this.device.queue.submit([commandEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
    }
};