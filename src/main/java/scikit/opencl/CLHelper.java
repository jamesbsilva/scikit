package scikit.opencl;

/**
*    @(#)   CLHelper
*/  

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLPlatform;
import com.jogamp.opencl.gl.CLGLContext;
import com.jogamp.opencl.util.CLPlatformFilters;
import java.util.ArrayList;
import java.util.HashMap;
import javax.media.opengl.GLContext;

/**
*       CLHelper helps with managing JOCL calls to do OpenCL 
*   enabled computations. Requires installation of OpenCL drivers and jocl / gluegen
*   library and its runtimes.
* 
* <br><br><br>
* 
* @see <a href="https://developer.nvidia.com/cuda-downloads">https://developer.nvidia.com/cuda-downloads</a> 
* @see <a href="http://developer.amd.com/tools-and-sdks/heterogeneous-computing/amd-accelerated-parallel-processing-app-sdk/downloads/">http://developer.amd.com/tools-and-sdks/heterogeneous-computing/amd-accelerated-parallel-processing-app-sdk/downloads/</a> 
* @see <a href="https://jogamp.org/jocl/www/">https://jogamp.org/jocl/www/</a> 
* 
* <br><br><br>
* 
*   Example Usage - <br> <br>
* 
* clhelper = new CLHelper();  <br>
* clhelper.initializeOpenCL("GPU");  <br>
* // 1000 float array  <br>
* int esize = 1000;  <br>
* int gsize = 1000;  <br>
* int lsize = 100;  <br>
* String kernel = "vector_iter";   <br>
* // Case with filename vector_iter.cl in a default search dir  <br>
* clhelper.createKernel(kernel);   <br>
* // float buffer for kernel   <br>
* clhelper.createFloatBuffer(kernel, 0, esize, 0.0f, 0);  <br>
* // integer argument for kernel  <br>
* clhelper.setIntArg(kernel, 0,esize);  <br>
* clhelper.runKernel(kernel, gsize, lsize);  <br>
* // get the first 3 values in the float array with buffer num 0 and print them while retrieving  <br>
* clhelper.getFloatBufferAsArray(kernel, 0, 3, true);  <br>
* 
* 
* <br>
* 
* @author      James B. Silva <jbsilva @ bu.edu>                 
* @since       2013
*/
public class CLHelper{
    private CLPlatform[] clPl;
    private CLContext context=null;
    private CLGLContext contextCLGL=null;
    private CLDevice device;
    private CLCommandQueue queue;
    private CLArgHelper arghandler;
    private CLKernelHelper kernelhandler;
    private boolean outputMode = true;
    private boolean glEnabled = false;
    
    public CLHelper(){
        kernelhandler = new CLKernelHelper();
    }
    
    /**
    *      addKernelSearchDir adds a directory to search for kernels in.
    * 
    * @param newdir - directory to add to search directories for kernels
    */
    public void addKernelSearchDir(String newdir){
        if(!newdir.endsWith("/")){newdir = newdir+"/";}
        kernelhandler.addKernelSearchDir(newdir);
    }
    
    // getters
    
    public boolean isGLenabled(){return glEnabled;}
        
    /**
    *      getContext gets the OpenCL context.
    * @return context
    */
    public CLContext getContext(){return context;}
        
    /**
    *      getContext gets the OpenCL context.
    * @return context
    */
    public CLGLContext getContextGLCL(){return contextCLGL;}
    
    /**
    *      getQueue gets the OpenCL command queue.
    * @return queue
    */
    public CLCommandQueue getQueue(){return queue;}
    /**
    *      getContext gets the OpenCL kernel argument types.
    * @return kernel argument types
    */
    public HashMap<String,ArrayList<String>> getArgTypes(){return kernelhandler.getArgTypes();}
    
    /**
    *      getArgHandler gets the OpenCL argument handler.
    * @return argument handler
    */
    public CLArgHelper getArgHandler(){return arghandler;}
    
    /**
    *      setOutputMode sets level of output by turning off/on some output strings.
    * 
    * @param md - true if outputting 
    */
    public void setOutputMode(boolean md){
        kernelhandler.setOutputMode(md);
        outputMode = md;
    }
    
    /**
    *   initializeOpenCL setups the OpenCL context to run the simulation in the device.
    * 
    * @param deviceType - string for specific device. "GPU","CPU","NVIDIA" example supported types
    *   "" for no preference.
    */
    public void initializeOpenCL(String deviceType){
        initializeOpenCL(deviceType, "",false,null);
    }
    
    
    /**
    *   initializeOpenCL setups the OpenCL context to run the simulation in the device.
    * 
    * @param deviceType - string for specific device. "GPU","CPU","NVIDIA" example supported types
    *   "" for no preference.
    */
    public void initializeOpenCL(String deviceType, boolean glenab,GLContext con){
        initializeOpenCL(deviceType, "",glenab,con);
    }
    
    /**
    *   initializeOpenCL setups the OpenCL context to run the simulation in the device.
    * 
    * @param deviceType - string for specific device. "GPU","CPU","NVIDIA" example supported types
    *   "" for no preference.
    * @param deviceName - device name if looking for specific device in platform
    */
    @SuppressWarnings("unchecked")
    public void initializeOpenCL(String deviceType,String deviceName, boolean glenab, GLContext con){
        outputClassStarLine();        
        glEnabled  = glenab;
           
        // search for platform support given device string
        try {
            if(deviceType.equalsIgnoreCase("GPU") || 
                deviceType.equalsIgnoreCase("Graphics Processor") ||
                deviceType.equalsIgnoreCase("Graphics")){
                clPl = CLPlatform.listCLPlatforms(CLPlatformFilters.type(CLDevice.Type.GPU));}
            else if(deviceType.equalsIgnoreCase("CPU") || 
                deviceType.equalsIgnoreCase("Processor")){
                clPl = CLPlatform.listCLPlatforms(CLPlatformFilters.type(CLDevice.Type.CPU));
            }else{
                clPl = CLPlatform.listCLPlatforms();
            }
        } catch (NoClassDefFoundError x) {
            x.printStackTrace();
        }
        
        if(deviceType.equalsIgnoreCase("NVIDIA")){
            deviceType = "NVIDIA CUDA";
        }
        
        if(deviceType.equalsIgnoreCase("")){
            outputClassString("List Platforms");
        }else{
            outputClassString("Searching Platforms For Device : " +deviceType);
        }
        
        int platformid = 0;
        int deviceid = 0;
        boolean deviceFound =false;
        
        // If looking for vendor specific type look for it otherwise list all devices
        for(int i =0;i < clPl.length;i++){
            CLDevice[] devices =clPl[i].listCLDevices();
            if(clPl[i].getName().equalsIgnoreCase(deviceType) || 
                    deviceType.equals("") || deviceType.equals("GPU") 
                    || deviceType.equals("CPU")){
            int j=0;
            while(!deviceFound && j<devices.length){
                if(devices[j].getName().equalsIgnoreCase(deviceName)){
                    deviceFound =true;
                    //printDeviceInfo(devices[j]);
                    deviceid=j;
                    platformid = i;
                }else if(deviceName.equals("")){
                    deviceFound =true;
                    // assert GPU type. necessary due to AMD APP mislabeling of Intel CPUs
                    if(deviceType.contains("GPU")){
                        deviceid = assertGPUIsRightType(i,glEnabled);
                        platformid = i;
                        if(deviceid>=0){deviceName ="truemaxGPU";}
                    }
                }else{j++;}
            }}       
        }
        if(!deviceFound){System.err.println("CLHelper | DEVICE ("+deviceName+")  NOT FOUND");}
    
        // set up (uses default CLPlatform and creates context for all devices)
        if(!glEnabled){
            context = CLContext.create(clPl[platformid]);
            // select fastest device
            if(deviceName.equalsIgnoreCase("")){
                device = context.getMaxFlopsDevice();
            }else{
                device = (context.getDevices())[deviceid];
            }
            queue = device.createCommandQueue();
        }else{
            // select fastest device
            if(deviceName.equalsIgnoreCase("")){
                device = clPl[platformid].getMaxFlopsDevice();
            }else{
                device = clPl[platformid].listCLDevices()[deviceid];
            }
            //System.out.println("Context: "+con+"    device: "+device);

            contextCLGL = CLGLContext.create(con, device);    
            queue = contextCLGL.getMaxFlopsDevice().createCommandQueue();
        }
        //out.println("created "+context);
    
        //out.println("using "+device);
        System.out.println("CLHelper | ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
        System.out.println("CLHelper | ++++++++++++++++  Using  +++  Device  +++  Number  +++++===  "+deviceid);
        
        printDeviceInfo(device);
        // create command queue on device.
        
        arghandler = new CLArgHelper(this);
        kernelhandler.setOpenCLReqs(queue,arghandler,glEnabled,context,contextCLGL);
        outputClassStarLine();
    }
    
    
    /**
    *       getCurrentDevice1DMaxWorkItems returns maximum workitems in 0th dimension.
    */
    public int getCurrentDevice1DMaxWorkItems(){
        if(device == null){
            System.err.println("CLHelper | CURRENT DEVICE NOT INTIALIZED.");
            return 64;
        }else{
            return device.getMaxWorkItemSizes()[0];
        }
    }
        
    /**
    *       listAllDevices list all devices of given device type.
    * 
    * @param deviceType - type of devices to list ie a CPU or GPU
    */
    @SuppressWarnings("unchecked")
    public void listAllDevices(String deviceType){
        outputClassStarLine(); 
         
        // search for platform support given device string
        if(deviceType.equalsIgnoreCase("GPU") || 
            deviceType.equalsIgnoreCase("Graphics Processor") ||
            deviceType.equalsIgnoreCase("Graphics")){
            clPl = CLPlatform.listCLPlatforms(CLPlatformFilters.type(CLDevice.Type.GPU));
        }else if(deviceType.equalsIgnoreCase("CPU") || 
            deviceType.equalsIgnoreCase("Processor")){
            clPl = CLPlatform.listCLPlatforms(CLPlatformFilters.type(CLDevice.Type.CPU));
        }else{
            clPl = CLPlatform.listCLPlatforms();
        }
        
        if(deviceType.equalsIgnoreCase("NVIDIA")){
            deviceType = "NVIDIA CUDA";
        }
        
        if(deviceType.equalsIgnoreCase("")){
            outputClassString("List All Platforms of "+clPl.length+"   platforms");
        }else{
            outputClassString("Searching Platforms For Device : " +deviceType);
        }
        
        System.out.println("CLHelper | ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
        System.out.println("CLHelper | ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
        // If looking for vendor specific type look for it otherwise list all devices
        for(int i =0;i < clPl.length;i++){
            CLDevice[] devices =clPl[i].listCLDevices();
            outputClassString("Platform: "+clPl[i].getName()+"    Devices Available on platform: "+devices.length);
            if(clPl[i].getName().equalsIgnoreCase(deviceType) || 
                    deviceType.equals("") || deviceType.equals("GPU") 
                    || deviceType.equals("CPU")){
                System.out.println("CLHelper | ***********************************************************************");
                for(int j = 0;j<devices.length;j++){
                    if(deviceType.equalsIgnoreCase("GPU") || 
                        deviceType.equalsIgnoreCase("Graphics Processor") ||
                        deviceType.equalsIgnoreCase("Graphics")){
                        if(isRealGPU(devices[j])){printDeviceInfo(devices[j]);}
                    }else if(deviceType.equalsIgnoreCase("CPU") || 
                                deviceType.equalsIgnoreCase("Processor")){
                        if(!isRealGPU(devices[j])){printDeviceInfo(devices[j]);}
                    }else{
                        printDeviceInfo(devices[j]);
                    }
                }
            }       
        }
    }
    
    /**
    *       assertGPUIsRightType asserts that if calling for a GPU the device with max flops
    *   is a GPU. This is a hack necessary because AMD's OpenCL implementation gives back Intel
    *   CPUs as being of type GPU.
    * 
    * @param type
    * @param i
    * @return proper max flops device
    */
    private int assertGPUIsRightType(int i, boolean glEnable){
        int j = 0;
        boolean rightType =false;
        // Flagging Intel and AMD as CPU
        if(clPl[i].getMaxFlopsDevice().getName().contains("Intel")
                ||clPl[i].getMaxFlopsDevice().getName().contains("Intel(R)")
                ||clPl[i].getMaxFlopsDevice().getName().contains("CPU")
                ||clPl[i].getMaxFlopsDevice().getName().contains("AMD")){
        }else{rightType=true;printMaxFlopsDeviceInfo(i);j=-11;}
        
        if(!rightType){
            // Search for correct max flops GPU
            CLDevice[] devices = clPl[i].listCLDevices();
            
            int currMaxFlops = 0;
            long currMaxGlobalMem = 0;
            int currMax = 0;
            int currFlops;
            long currGlobalMem=0;
            
            for(int current = 0;current < devices.length;current++){
                if(devices[current].getName().contains("AMD")
                        ||devices[current].getName().contains("Intel(R)")
                        ||devices[current].getName().contains("CPU")
                        ||devices[current].getName().contains("Intel")){
                }else{
                    // if looking for gl sharing device skip incompatible device
                    if(glEnable && !devices[current].isGLMemorySharingSupported()){
                        continue;
                    }
                    currFlops = devices[current].getMaxClockFrequency()*devices[current].getMaxComputeUnits();
                    currGlobalMem = devices[current].getMaxClockFrequency()*devices[current].getGlobalMemSize();
                    if(currFlops > currMaxFlops){
                        currMaxFlops =currFlops;
                        currMaxGlobalMem = currGlobalMem;
                        currMax = current;
                    }else if(currFlops == currMaxFlops && currMaxGlobalMem < currGlobalMem){
                        currMaxFlops =currFlops;
                        currMaxGlobalMem = currGlobalMem;
                        currMax = current;
                    }
                }
            }
            j= currMax;
        }
        return j;
    }
    
    /**
    *       isRealGPU asserts that if calling for a GPU it 
    *   is a GPU. This is a hack necessary because AMD's OpenCL implementation gives back Intel
    *   CPUs as being of type GPU.
    * 
    * @param dev - device
    * @return true if gpu
    */
    private boolean isRealGPU(CLDevice dev){
        boolean rightType =false;
        if(dev.getName().contains("AMD")
                    ||dev.getName().contains("Intel(R)")
                    ||dev.getName().contains("CPU")
                    ||dev.getName().contains("Intel")){
        }else{rightType = true;}        
        return rightType;
    }
    
    /**
    *       printMaxFlopsDeviceInfo prints information for max flops device.
    * 
    * @param i - platform for which to print information 
    */
    private void printMaxFlopsDeviceInfo(int i){
            outputClassString(clPl[i].getName());
            outputClassString("Devices per platform :"+clPl[i].listCLDevices().length);
            outputClassString("Compute Devices on max flops device:"+
                clPl[i].getMaxFlopsDevice().getMaxComputeUnits());
            outputClassString("Max Clock Frequency device:"+
                clPl[i].getMaxFlopsDevice().getMaxClockFrequency());
            outputClassString("Max global memory for device:"+
                (clPl[i].getMaxFlopsDevice().getGlobalMemSize()/1000000)+" MB");
            outputClassString("Max local memory for device:"+
                (clPl[i].getMaxFlopsDevice().getLocalMemSize()/1000)+" KB");
            outputClassString("Max memory allocateable size for device:"+
                (clPl[i].getMaxFlopsDevice().getMaxMemAllocSize()/1000000)+" MB");
    }
    
    /**
    *       printDeviceInfo prints information for device.
    * 
    * @param dev - device for which to print information 
    */
    private void printDeviceInfo(CLDevice dev){
            outputClassString(dev.getName());
            outputClassString("Compute Units on max flops device:"+
                dev.getMaxComputeUnits());
            outputClassString("Max Clock Frequency on device:"+
                dev.getMaxClockFrequency());
            outputClassString("Max global memory for device:"+
                (dev.getGlobalMemSize()/1000000)+" MB");
            outputClassString("Max local memory for device:"+
                (dev.getLocalMemSize()/1000)+" KB");
            outputClassString("Max memory allocateable size for device:"+
                (dev.getMaxMemAllocSize()/1000000)+" MB");
            outputClassString("Max workgroup size: "+
                dev.getMaxWorkGroupSize());
            outputClassString("Max workitems in 1d:"+
                dev.getMaxWorkItemSizes()[0]);
            outputClassString("________________________________");
    }
    
    /**
    *       closeOpenCL releases the OpenCL context.
    */
    public void closeOpenCL(){
        outputClassString("Closing OpenCL by releasing context.");
        if(glEnabled){
            contextCLGL.release();
        }else{
            context.release();
        }
    }

    
    //***************************************************************************
    //************Kernel***Handler****Methods**********************************
    //***************************************************************************
        
    /**
    *   createKernel builds the kernel in the OpenCL device.
    * 
    * @param kernelname - kernel name -needs to have filename with same name
    */
    public void createKernel(String kernelname){
        kernelhandler.createKernel("", kernelname);
    }
     
    /**
    *   createKernel builds the kernel in the OpenCL device.
    * 
    * @param fname - kernel file name- null for device filename is the same 
    *       as kernelname but cl as filetype
    * @param kernelname - kernel name 
    */
    public void createKernel(String fname,String kernelname){
        kernelhandler.createKernel(fname, kernelname);
    }
    
    /**
    *       createKernelFromSource builds the kernel in the OpenCL device 
    *   from given source.
    * 
    * @param kernelname - kernel name source
    * @param source - kernel source
    */
    public void createKernelFromSource(String kernelname,String source){
        kernelhandler.createKernelFromSource(kernelname, source);
    }
    
    /**
    *       runKernel runs the kernel in the OpenCL device.
    * 
    * @param kernelname - kernel name of the kernel to be run
    * @param gsize - global work size
    * @param lsize - local work size
    */
    public void runKernel(String kernelname, int gsize,int lsize){
        kernelhandler.runKernel(kernelname, gsize, lsize);
    }
    
    /**
    *       runKernel2D runs the kernel with 2d  parameters in the OpenCL device.
    * 
    * @param kernelname - kernel name of the kernel to be run
    * @param gsizex - global work size
    * @param lsizex - local work size
    * @param gsizey - global work size 2nd dimension
    * @param lsizey - local work size 2nd dimension
    */
    public void runKernel2D(String kernelname, int gsizex, int gsizey, int lsizex, int lsizey){
        kernelhandler.runKernel2D(kernelname, gsizex, gsizey, lsizex, lsizey);
    }
   
     
    /**
    *       runKernel3D runs the kernel with 3d  parameters in the OpenCL device.
    * 
    * @param kernelname - kernel name of the kernel to be run
    * @param gsizex - global work size
    * @param lsizex - local work size
    * @param gsizey - global work size 2nd dimension
    * @param lsizey - local work size 2nd dimension
    * @param gsizez - global work size 3rd dimension
    * @param lsizez - local work size 3rd dimension
    */
    public void runKernel3D(String kernelname, int gsizex,int lsizex,int gsizey,int lsizey, int gsizez, int lsizez){
        kernelhandler.runKernel3D(kernelname, gsizex, lsizex, gsizey, lsizey, gsizez, lsizez);
    }
    
    
    /**
    *       setQueueBuff sets the buffers in the OpenCL queue so the device knows  
    *   which buffers to access.
    * 
    * @param kernelname - kernel which needs buffers in queue
    */
    public void setQueueBuff(String kernelname){
        kernelhandler.setQueueBuff(kernelname);
    }
    
    /**
    *       setKernelArg sets the arguments of the given kernel by using 
    *   the arguments parsed from source.
    * 
    * @param kernelname - kernel to set arguments for
    */
    public void setKernelArg(String kernelname){
        setKernelArg(kernelname,false);
    }
    
    /**
    *       setKernelArg sets the arguments of the given kernel by using 
    *   the arguments parsed from source.
    * 
    * @param kernelname - kernel to set arguments for
    * @param setPrev - true if already set previously
    */
    public void setKernelArg(String kernelname, boolean setPrev){
        kernelhandler.setKernelArg(kernelname,setPrev);
    }
    
    /**
    *       getDeviceUsedMB determines the amount of MB used by the kernel
    *   in the OpenCL device
    * 
    * 
    * @param kernelname - kernel name 
    * @return MB used by kernel
    */
    public int getDeviceUsedMB(String kernelname){
        return kernelhandler.getDeviceUsedMB(kernelname);
    }
    
    //***************************************************************************
    //***************************************************************************
    //***************************************************************************
    //************Argument***Handler****Methods**********************************
    //***************************************************************************
    //***************************************************************************
    //***************************************************************************
    //***************************************************************************
    //***************************************************************************
    //***************************************************************************
    //***************************************************************************
    //***************************************************************************
    
    
    /**
    *       createIntBuffer creates and fills an integer buffer then adds it to 
    *   the list of int buffers for this kernel at given argument entry number. 
    * 
    * 
    * @param kernelname - kernel to add buffer for
    * @param argn - argument type entry number
    * @param size - size of buffer
    * @param s0 - initial values for the buffer 
    * @param readwrite - int (2,1,0) - 2 write only, 1 read only, 0 read and write
    */
    public void createIntBuffer(String kernelname,int argn,int size, int[] s0,int readwrite){
        ArrayList<Integer> push = new ArrayList<Integer>();
        for(int i = 0;i<s0.length;i++){
            push.add(i,s0[i]);
        }
        arghandler.manageBuffer(kernelname,"int",argn,push,convertRWIntToString(readwrite));
    }
    
    /**
    *       createIntBuffer creates and fills an integer buffer then adds it to 
    *   the list of int buffers for this kernel at given argument entry number. 
    * 
    * 
    * @param kernelname - kernel to add buffer for
    * @param argn - argument type entry number
    * @param size - size of buffer
    * @param s0 - initial values for the buffer 
    * @param readwrite - int (2,1,0) - 2 write only, 1 read only, 0 read and write
    */
    public void createIntBuffer(String kernelname,int argn,int size, int s0,int readwrite){
        arghandler.manageBuffer(kernelname,s0,size,convertRWIntToString(readwrite),argn);
    }
    
    /**
    *       createIntBuffer creates and fills an integer buffer then adds it to 
    *   the list of int buffers for this kernel at given argument entry number. 
    * 
    * 
    * @param kernelname - kernel to add buffer for
    * @param argn - argument type entry number
    * @param size - size of buffer
    * @param s0 - initial values for the buffer 
    * @param readwrite - int (2,1,0) - 2 write only, 1 read only, 0 read and write
    */
    public void createIntBuffer(String kernelname,int argn,int size, int s0,String readwrite){
        arghandler.manageBuffer(kernelname,s0,size,readwrite,argn);
    }
    
    /**
    *       createIntBuffer creates and fills an integer buffer then adds it to 
    *   the list of int buffers for this kernel at given argument entry number. 
    * 
    * 
    * @param kernelname - kernel to add buffer for
    * @param argn - argument type entry number
    * @param size - size of buffer
    * @param s0 - initial values for the buffer 
    * @param readwrite - int (2,1,0) - 2 write only, 1 read only, 0 read and write
    * @param modFill - fill using a modulus fill which fills based on mod of array size
    */
    public void createIntBuffer(String kernelname,int argn,int size, int[] s0,int readwrite,boolean modFill){
        ArrayList<Integer> push = new ArrayList<Integer>();
        for(int i = 0;i<s0.length;i++){
            push.add(i,s0[i]);
        }
        arghandler.manageBuffer(kernelname,"int",argn,push,size,convertRWIntToString(readwrite),modFill);
    }
    
    public void createLongBuffer(String kernelname,int argn,int size, long[] s0,int readwrite,boolean modFill){
        ArrayList<Long> push = new ArrayList<Long>();
        for(int i = 0;i<s0.length;i++){
            push.add(i,s0[i]);
        }
        arghandler.manageBuffer(kernelname,"long",argn,push,size,convertRWIntToString(readwrite),modFill);
    }

        /**
    *       createIntBuffer creates and fills an integer buffer then adds it to 
    *   the list of int buffers for this kernel at given argument entry number. 
    * 
    * 
    * @param kernelname - kernel to add buffer for
    * @param argn - argument type entry number
    * @param size - size of buffer
    * @param s0 - initial values for the buffer 
    * @param readwrite - int (2,1,0) - 2 write only, 1 read only, 0 read and write
    * @param modFill - fill using a modulus fill which fills based on mod of array size
    */
    public void createIntBuffer(String kernelname,int argn,int size, ArrayList<Integer> s0,String readwrite,boolean modFill){
        arghandler.manageBuffer(kernelname,"int",argn,s0,size,readwrite,modFill);
    }

    
    
    /**
    *       createFloatBuffer creates and fills an float buffer then adds it to 
    *   the list of float buffers for this kernel at given argument entry number. 
    * 
    * 
    * @param kernelname - kernel to add buffer for
    * @param argn - argument type entry number
    * @param size - size of buffer
    * @param s0 - initial value for the buffer entries 
    * @param readwrite - int (2,1,0) - 2 write only, 1 read only, 0 read and write
    */    
    public void createFloatBuffer(String kernelname,int argn,int size, float s0,int readwrite){
        arghandler.manageBuffer(kernelname,s0,size,convertRWIntToString(readwrite),argn);
    }    


    /**
    *       createFloatBuffer creates and fills an float buffer then adds it to 
    *   the list of float buffers for this kernel at given argument entry number. 
    * 
    * 
    * @param kernelname - kernel to add buffer for
    * @param argn - argument type entry number
    * @param size - size of buffer
    * @param s0 - initial value for the buffer entries 
    * @param readwrite - int (2,1,0) - 2 write only, 1 read only, 0 read and write
    */    
    public void createFloatBuffer(String kernelname,int argn,int size, float s0,String readwrite){
        arghandler.manageBuffer(kernelname,s0,size,readwrite,argn);
    }
    
    
    /**
    *       createFloatBuffer creates a buffer then adds it to 
    *   the list of float buffers for this kernel at given argument entry number. 
    * 
    * 
    * @param kernelname - kernel to add buffer for
    * @param argn - argument type entry number
    * @param size - size of buffer
    * @param readwrite - int (2,1,0) - 2 write only, 1 read only, 0 read and write
    */    
    public void createFloatBuffer(String kernelname,int argn,int size, int readwrite){
        arghandler.manageBuffer(kernelname,"float",argn,size,convertRWIntToString(readwrite));
    }    
    
    
    /**
    *       createFloatBuffer creates and fills an float buffer then adds it to 
    *   the list of float buffers for this kernel at given argument entry number. 
    * 
    * 
    * @param kernelname - kernel to add buffer for
    * @param argn - argument type entry number
    * @param size - size of buffer
    * @param s0 - initial values for the buffer 
    * @param readwrite - int (2,1,0) - 2 write only, 1 read only, 0 read and write
    * @param modFill - fill using a modulus fill which fills based on mod of array size
    */    
    public void createFloatBuffer(String kernelname,int argn,int size, float[] s0,int readwrite, boolean modFill){
        ArrayList<Float> push = new ArrayList<Float>();
        for(int i = 0;i<s0.length;i++){
            push.add(i,s0[i]);     
        }
        arghandler.manageBuffer(kernelname,"float",argn,push,size,convertRWIntToString(readwrite),modFill);
    }
    public void createFloatBuffer(String kernelname,int argn,int size, float[] s0,String readwrite, boolean modFill){
        ArrayList<Float> push = new ArrayList<Float>();
        for(int i = 0;i<s0.length;i++){
            push.add(i,s0[i]);     
        }
        arghandler.manageBuffer(kernelname,"float",argn,push,size,readwrite,modFill);
    }
    
    /**
    *       createFloatBuffer creates and fills an float buffer then adds it to 
    *   the list of float buffers for this kernel at given argument entry number. 
    * 
    * 
    * @param kernelname - kernel to add buffer for
    * @param argn - argument type entry number
    * @param size - size of buffer
    * @param s0 - initial values for the buffer 
    * @param readwrite - int (2,1,0) - 2 write only, 1 read only, 0 read and write
    * @param modFill - fill using a modulus fill which fills based on mod of array size
    */    
    public void createFloatBuffer(String kernelname,int argn,int size, ArrayList<Float> s0,String readwrite, boolean modFill){
        arghandler.manageBuffer(kernelname,"float",argn,s0,size,readwrite,modFill);
    }
    
    /**
    *       setFloatBuffer sets a buffers values to the given array list of values at the given kernel buffer.  
    * 
    * @param kernelname - kernel to add buffer for
    * @param argn - argument type entry number
    * @param size - size of buffer
    * @param s0 - initial values for the buffer 
    * @param modFill - fill using a modulus fill which fills based on repetitions of array list
    */    
    public void setFloatBuffer(String kernelname,int argn,int size, ArrayList<Float> s0, boolean modFill){
        if(modFill){    
            arghandler.manageBuffer(kernelname, "float", argn, "set", size, "", 2, s0, null,false,0);
        }else{arghandler.manageBuffer(kernelname, "float", argn, "set", size, "", 1, s0, null,false,0);}  
    }
    

    /**
    *       setIntBuffer sets a buffers values to the given array list of values at the given kernel buffer.  
    * 
    * @param kernelname - kernel to add buffer for
    * @param argn - argument type entry number
    * @param size - size of buffer
    * @param s0 - initial values for the buffer 
    * @param modFill - fill using a modulus fill which fills based on repetitions of array list
    */    
    public void setIntBuffer(String kernelname,int argn,int size, ArrayList<Integer> s0, boolean modFill){
        if(modFill){arghandler.manageBuffer(kernelname, "int", argn, "set", size, "", 2, s0, null,false,0);}
        else{arghandler.manageBuffer(kernelname, "int", argn, "set", size, "", 1, s0, null,false,0);}
    }

    /**
    *       setLongBuffer sets a buffers values to the given array list of values at the given kernel buffer.  
    * 
    * @param kernelname - kernel to add buffer for
    * @param argn - argument type entry number
    * @param size - size of buffer
    * @param s0 - initial values for the buffer 
    * @param modFill - fill using a modulus fill which fills based on repetitions of array list
    */    
    public void setLongBuffer(String kernelname,int argn,int size, ArrayList<Long> s0, boolean modFill){
        if(modFill){    
        arghandler.manageBuffer(kernelname, "long", argn, "set", size, "", 2, s0, null,false,0);}
        else{arghandler.manageBuffer(kernelname, "long", argn, "set", size, "", 1, s0, null,false,0);}
        
    }
    
    /**
    *       manageBuffer main class which allows for managing of buffers of long,
    *   float, or int type.
    * 
    * 
    * @param kernelname - kernel to add buffer for
    * @param type - type for argument (int, long, or float)
    * @param argn - argument type entry number
    * @param updateMode - String for action of buffer to do (create,set,get)
    * @param size - size of buffer
    * @param readwrite - "r" for read , "w" for write, else default read/write.
    * @param fill - 1 if filling buffer regular, 2 if fill based on repeating array list
    * @param s0 - initial value array of all buffer entries
    * @param  inbuff - buffer to set argument to
    * @return created buffer
    */
    public CLBuffer manageBuffer(String kernelname, String type, int argn, String updateMode, int size,
            String readwrite, int fill , ArrayList s0, boolean fillMod, CLBuffer inbuff){
        return arghandler.manageBuffer(kernelname, type, argn, updateMode, size, readwrite, fill, s0,  inbuff,false,0);
    }
    
    /**
    *       copyFLBufferAcrossKernel copies a buffer from source kernel float buffer list
    *   into destination kernel float buffer list.
    * 
    * @param skernel - source kernel name
    * @param sargn - source kernel float argument number
    * @param dkernel - destination kernel name
    * @param dargn - destination kernel float argument number
    */
    public void copyFlBufferAcrossKernel(String skernel,int sargn,String dkernel,int dargn){
        arghandler.copyBufferAcrossKernel("float",skernel,  sargn, dkernel, dargn);
    }
    
    /**
    *       copyFLBufferAcrossKernel copies a buffer from source kernel float buffer list
    *   into destination kernel float buffer list.
    * 
    * @param skernel - source kernel name
    * @param sargn - source kernel float argument number
    * @param dkernel - destination kernel name
    * @param dargn - destination kernel float argument number
    * @param setMode - true if setting buffers instead of creating
    */
    public void copyFlBufferAcrossKernel(String skernel,int sargn,String dkernel,int dargn, boolean setMode){
        arghandler.copyBufferAcrossKernel("float",skernel,  sargn, dkernel, dargn, setMode);
    }
    
    /**
    *       copyIntBufferAcrossKernel copies a buffer from source kernel int buffer list
    *   into destination kernel int buffer list.
    * 
    * @param skernel - source kernel name
    * @param sargn - source kernel int argument number
    * @param dkernel - destination kernel name
    * @param dargn - destination kernel int argument number
    */
    public void copyIntBufferAcrossKernel(String skernel,int sargn,String dkernel,int dargn){
        arghandler.copyBufferAcrossKernel("int",skernel,  sargn, dkernel, dargn);
    }
    
    /**
    *       setIntArg creates an int and adds it to the arguments for kernel.
    * 
    * @param kernelname - name of kernel to add argument to list
    * @param argn - argument type entry number
    * @param val - initial value of int
    */
    public void setIntArg(String kernelname,int argn, int val){
        arghandler.setIntArg(kernelname,argn,val);
        if(kernelhandler.getKernelArgStatus(kernelname)){kernelhandler.setKernelArg(kernelname, true);};
    }

    /**
    *       setLongArg creates an Long and adds it to the arguments for kernel.
    * 
    * @param kernelname - name of kernel to add argument to list
    * @param argn - argument type entry number
    * @param val - initial value of Long
    */
    public void setLongArg(String kernelname,int argn, long val){
        arghandler.setLongArg(kernelname,argn,val);
        if(kernelhandler.getKernelArgStatus(kernelname)){kernelhandler.setKernelArg(kernelname, true);};
    }

    
    /**
    *       getIntArg get an int from the kernel set.
    * 
    * @param kernelname - name of kernel to get int from
    * @param argn - argument type entry number
    */
    public int getIntArg(String kernelname,int argn){
        return arghandler.getIntArgs().get(kernelname).get(argn);
    }
    
    /**
    *       getLongArg get a long from the kernel set.
    * 
    * @param kernelname - name of kernel to get long from
    * @param argn - argument type entry number
    */
    public long getLongArg(String kernelname,int argn){
        return arghandler.getLongArgs().get(kernelname).get(argn);
    }
    
    /**
    *       getFloatArg get a float from the kernel set.
    * 
    * @param kernelname - name of kernel to get float from
    * @param argn - argument type entry number
    */
    public float getFloatArg(String kernelname,int argn){
        return arghandler.getFloatArgs().get(kernelname).get(argn);
    }
    
    /**
    *       createFloatArg creates an float and adds it to the arguments for kernel.
    * 
    * @param kernelname - name of kernel to add argument to list
    * @param argn - argument type entry number
    * @param val - initial value of float
    */
    public void setFloatArg(String kernelname,int argn, float val){
        arghandler.setFloatArg(kernelname,argn,val); 
        if(kernelhandler.getKernelArgStatus(kernelname)){kernelhandler.setKernelArg(kernelname, true);};
    }
    
    /**
    *           getIntBufferAsArray retrieves the device buffer and returns it
    *   into an array which can be used in host.
    * 
    * @param kernelname - kernel to retrieve buffer from
    * @param argn - argument number 
    * @param size - size of buffer to retrieve 
    * @param print - true if printing out retrieved values
    * @return integer array version of buffer
    */
    public int[] getIntBufferAsArray(String kernelname,int argn, int size,boolean print){
        return arghandler.getIntBufferAsArray(kernelname, argn, size, print);
    }

    /**
    *           getFloatBufferAsArray retrieves the device buffer and returns it
    *   into an array which can be used in host.
    * 
    * @param kernelname - kernel to retrieve buffer from
    * @param argn - argument number 
    * @param size - size of buffer to retrieve 
    * @param print - true if printing out retrieved values
    * @return float array version of buffer
    */
    public float[] getFloatBufferAsArray(String kernelname,int argn, int size,boolean print){
        return arghandler.getFloatBufferAsArray(kernelname, argn, size, print);
    }
    
    /**
    *           getBufferAsArrayList retrieves the device buffer and returns it
    *   into an array list which can be used in host.
    * 
    * @param kernelname - kernel to retrieve buffer from
    * @param type - type for argument (int, long, or float)
    * @param argn - argument number 
    * @param size - size of buffer to retrieve 
    * @param print - true if printing out retrieved values
    * @return integer array version of buffer
    */
    public ArrayList getBufferAsArrayList(String kernelname,String type ,int argn, int size,boolean print){
        return arghandler.getBufferAsArrayList(kernelname, type, argn, size, print);
    }
    
    /**
    *           getBufferIntAsArrayList retrieves the device buffer and returns it
    *   into an array list int which can be used in host.
    * 
    * @param kernelname - kernel to retrieve buffer from
    * @param argn - argument number 
    * @param size - size of buffer to retrieve 
    * @param print - true if printing out retrieved values
    * @return integer array version of buffer
    */
    public ArrayList<Integer> getBufferIntAsArrayList(String kernelname ,int argn, int size,boolean print){
        // Dont need type checking because error will be thrown anyways
        @SuppressWarnings("unchecked")
        ArrayList<Integer> iarr = (ArrayList<Integer>)(arghandler.getBufferAsArrayList(kernelname, "int", argn, size, print));
        return iarr;
    }
    /**
    *           getBufferIntAsArrayList retrieves the device buffer and returns it
    *   into an array list int which can be used in host.
    * 
    * @param kernelname - kernel to retrieve buffer from
    * @param argn - argument number 
    * @param size - size of buffer to retrieve 
    * @param print - true if printing out retrieved values
    * @return integer array version of buffer
    */
    public ArrayList<Float> getBufferFlAsArrayList(String kernelname ,int argn, int size,boolean print){
        // Dont need type checking because error will be thrown anyways
        @SuppressWarnings("unchecked")
        ArrayList<Float> flarr = arghandler.getBufferAsArrayList(kernelname, "float", argn, size, print);
        return flarr;
    }
    
    /**
    *           getBufferSumAsDouble sums up all the elements of the queried buffer
    *   and returns as a double. Not optimally fast.
    * 
    * @param kernelname - kernel to retrieve buffer from
    * @param type - type for argument (int, long, or float)
    * @param argn - argument number 
    * @param size - size of buffer to retrieve 
    * @return double representing sum of contributions
    */
    public double getBufferSumAsDouble(String kernelname,String type ,int argn, int size){
        return arghandler.getBufferSumAsDouble(kernelname, type, argn, size);
    }
    
    /**
    *           quickGetLastBufferMax returns double of the maximum of 
    *   the last buffer that has been retrieved.
    * 
    * @return double of the maximum of the last buffer that has been retrieved 
    */
    public double quickGetLastBufferMax(){
        return arghandler.quickGetLastBufferMax();
    }
    
    
    
    /**
    *       createFloatBuffer creates and fills an float buffer then adds it to 
    *   the list of float buffers for this kernel at given argument entry number. 
    * 
    * 
    * @param kernelname - kernel to add buffer for
    * @param argn - argument type entry number
    * @param size - size of buffer
    * @param s0 - initial values for the buffer 
    * @param readwrite - int (2,1,0) - 2 write only, 1 read only, 0 read and write
    * @param modFill - fill using a modulus fill which fills based on mod of array size
    */    
    public void createFloatBufferGL(String kernelname,int argn,int size, float[] s0,String readwrite, boolean modFill, int glInd){
        ArrayList<Float> push = new ArrayList<Float>();
        for(int i = 0;i<s0.length;i++){
            push.add(i,s0[i]);     
        }
        int fill;
        if(modFill){fill = 2;}else{fill = 1;}
        int rw = convertStringToRWInt(readwrite);
        arghandler.manageBuffer(kernelname,"float",argn,"create",size,convertRWIntToString(rw),fill,push,null,true,glInd);
    }
    
    /**
    *       convertRWIntToString converts an integer read write key to a string version
    * 
    * @param readwrite - integer key
    * @return string key
    */
    private String convertRWIntToString(int readwrite){        
        if(readwrite==2){
            return "w";
        }else if(readwrite==1){
            return "r";
        }else if (readwrite==0){
            return "rw";
        }else{
            System.err.println("CLHelper | BAD READ WRITE INT: "+readwrite);
            return "";
        }
        
        
        
    }
    
    /**
    *       setPrintMode sets the print mode in getBuffer functions
    * 
    * @param md - true if printing just values false if print index and values 
    */
    public void setPrintMode(boolean md){
        arghandler.setPrintMode(md);
    }
    
    
    
    
    //***************************************************************************
    //***************************************************************************
    //***************************************************************************
    //************Helper****Methods**********************************************
    //***************************************************************************
    //***************************************************************************
    //***************************************************************************
    //***************************************************************************
    
    /**
    *       convertRWIntToString converts an integer read write key to a string version
    * 
    * @param readwrite - integer key
    * @return string key
    */
    public int convertStringToRWInt(String readwrite){
        int rw = 0;
        boolean rEnable = false;;
        boolean wEnable = false;
        if(readwrite.contains("Read")|| readwrite.contains("READ")||readwrite.contains("read")){
            rEnable = true;
        }else if(readwrite.contains("Write")|| readwrite.contains("WRITE")||readwrite.contains("write")){
            wEnable = true;
        }
        if(!(wEnable && rEnable)){
            if(wEnable){rw = 2;}
            if(rEnable){rw = 1;}
        }
        return rw;
    }
   
    /**
    *       maxLocalSize1D determines the largest local work size that is still
    *   allowed on current device. 
    * 
    * @param gsize- global work size
    * @return local work size
    */
    public int maxLocalSize1D(int gsize){
        int max = getCurrentDevice1DMaxWorkItems();
        int lsize = 1;
        if( gsize > max){
            lsize = max;
            while( gsize % lsize != 0 && lsize > 1 ){
                lsize--;
            }
        }else{
            lsize = gsize;
        }
        return lsize;
    } 

    /**
    *       maxLocalSize2D determines the largest local work size that is still
    *   allowed on current device. 
    * 
    * @param gsize- global work size
    * @return local work size
    */
    public int maxLocalSize2D(int gsize){
        int max = (int) Math.pow(getCurrentDevice1DMaxWorkItems(), 1.0/2.0);
        int lsize = 1;
        if( gsize > max){
            lsize = max;
            while( gsize % lsize != 0 && lsize > 1 ){
                lsize--;
            }
        }else{
            lsize = (int) Math.pow(gsize, 1.0/2.0);
        }
        return lsize;
    } 

    /**
    *       maxLocalSize3D determines the largest local work size that is still
    *   allowed on current device. 
    * 
    * @param gsize- global work size
    * @return local work size
    */
    public int maxLocalSize3D(int gsize){
        int max = (int) Math.pow(getCurrentDevice1DMaxWorkItems(), 1.0/3.0);
        int lsize = 1;
        if( gsize > max){
            lsize = max;
            while( gsize % lsize != 0 && lsize > 1 ){
                lsize--;
            }
        }else{
            lsize = (int) Math.pow(gsize, 1.0/3.0);
        }
        return lsize;
    } 

    private void outputClassString(String msg){
        if(outputMode){System.out.println("CLHelper | "+msg);}
    }
    private void outputClassStarLine(){
        outputClassString("***********************************************************************");
    }    
    private void outputClassDashLine(){
        outputClassString("-------------------------------------------------------------------------------");
    }
}