package scikit.opencl;

/**
* 
*    @(#)   CLKernelHelper
*/  

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;
import com.jogamp.opencl.gl.CLGLBuffer;
import com.jogamp.opencl.gl.CLGLContext;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;

/**
*          CLKernelHelper is a helper class for CLHelper that deals with kernel
*  operations.
* 
* 
* @author James B. Silva (jbsilva@bu.edu)
* @since       2013-07
*/
public class CLKernelHelper {
    private CLCommandQueue queue;
    private CLArgHelper arghandler;
    private CLContext context=null;
    private CLGLContext contextCLGL=null;
    private ArrayList<String> defaultKernelDirs;
    private HashMap<String,CLKernel> kernels;
    private HashMap<String,Boolean> kernelArgStatus;
    private HashMap<String,ArrayList<String>> argTypes;
    private boolean outputMode = true;
    private boolean glEnabled = false;

    public CLKernelHelper(){
        defaultKernelDirs = new ArrayList<String>();
        //default directories
        String curr = System.getProperty("user.dir");
        addKernelSearchDir(curr+"/GPUKernels/");
        addKernelSearchDir(curr+"/src/GPUKernels/");
        // Initialize all list and maps
        kernels = new HashMap<String,CLKernel>();
        kernelArgStatus = new HashMap<String,Boolean>();
        argTypes= new HashMap<String,ArrayList<String>>();
    }
    /**
    *      addKernelSearchDir adds a directory to search for kernels in.
    * 
    * @param newdir - directory to add to search directories for kernels
    */
    public void addKernelSearchDir(String newdir){
        defaultKernelDirs.add(formatDir(newdir));
        String[] files = (new File(newdir)).list();
        if(files != null){
            // add sub dirs one level down
            for(String name : files){
                if (new File(newdir+ name).isDirectory()){
                    defaultKernelDirs.add(formatDir(newdir+name));
                }
            }
        }
    }
    // format directory
    private String formatDir(String posdir){
        if(!posdir.endsWith("/")){posdir = posdir+"/";}
        return posdir;
    }
    
    /**
    *      setOpenCLReqs sets the command queue for the device which will run 
    *   OpenCL kernels
    * 
    * @param qu - command queue of OpenCL device
    * @param arg - CLArgHelper of OpenCL device
    * @param gl - boolean of gl enabled status of OpenCL device
    * @param con - context of OpenCL device
    * @param cong - context CLGL  queue of OpenCL device
    */
    public void setOpenCLReqs(CLCommandQueue qu, CLArgHelper arg, boolean gl, CLContext con, CLGLContext cong){
        queue = qu;
        context = con;
        contextCLGL = cong;
        glEnabled = gl;
        arghandler = arg;
    }
    
    /**
    *      getContext gets the OpenCL kernel argument types.
    * @return kernel argument types
    */
    public HashMap<String,ArrayList<String>> getArgTypes(){return argTypes;}
      
    /**
    *   createKernel builds the kernel in the OpenCL device.
    * 
    * @param fname - kernel file name- null for device filename is the same 
    *       as kernelname but cl as filetype
    * @param kernelname - kernel name 
    */
    public void createKernel(String fname,String kernelname){
        // Find kernel file
        if(fname==null || fname.equals("")){fname= kernelname+".cl";}
        String fname1 = defaultKernelDirs.get(0)+fname;
        File f = new File(fname1);
        int searchInd = 0;
        while( searchInd < defaultKernelDirs.size()){
            //outputClassString("Searching For Kernel : "+fname1);
            if(f.exists()){
                fname = fname1;
                break;
            }
            searchInd++;
            fname1 = (searchInd < defaultKernelDirs.size()) ? defaultKernelDirs.get(searchInd)+fname : fname1;
            f = new File(fname1);
        }
        if(!f.exists()){outputClassString("Kernel File for "+kernelname+" Not Found");}
        outputClassDashLine();
        outputClassString("Creating kernel:"+kernelname+"  from file: "+fname);
        // Read source code
        try{
            String sourceCode = readFile(fname);
            createKernelFromSource(kernelname,sourceCode);
        }catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
    *       createKernelFromSource builds the kernel in the OpenCL device 
    *   from given source.
    * 
    * @param kernelname - kernel name source
    * @param source - kernel source
    */
    public void createKernelFromSource(String kernelname,String source){
        // Read source code
        CLProgram program=null;
        argTypes.put(kernelname,getKernelIOTypes(kernelname,source));
        // load sources, create and build program
        if(glEnabled){
            program = contextCLGL.createProgram(source).build();
        }else{
            program = context.createProgram(source).build();
        }
        //outputClassString("prog: "+program);
        // kernel comes labeled therefore just push into kernels
        kernels.put(kernelname,program.createCLKernel(kernelname));
        kernelArgStatus.put(kernelname,false);
        outputClassDashLine();
    }
    
    /**
    *       runKernel runs the kernel in the OpenCL device.
    * 
    * @param kernelname - kernel name of the kernel to be run
    * @param gsize - global work size
    * @param lsize - local work size
    */
    public void runKernel(String kernelname, int gsize,int lsize){
        // Assert kernel buffers are setup 
        //assertKernelBuffersMade(kernelname);
        if(!kernelArgStatus.get(kernelname)){
            System.err.println("CLKernelHelper | Kernel Arguments for "+kernelname+" not previously set. Will Attempt to set arguments now. ");
            setKernelArg(kernelname);
        }
        setQueueBuff(kernelname);
        queue.put1DRangeKernel(kernels.get(kernelname), 0, gsize, lsize);
        queueOutputBuffers(kernelname);
        queue.finish();
    }
    
    /**
    *       runKernel2D runs the kernel with 2d  parameters in the OpenCL device.
    * 
    *  @param kernelname - kernel name of the kernel to be run
    *  @param gsizex - global work size
    *  @param lsizex -local work size
    *  @param gsizey - global work size 2nd dimension
    *  @param lsizey -local work size 2nd dimension
    */
    public void runKernel2D(String kernelname, int gsizex, int gsizey, int lsizex, int lsizey){
        // Assert kernel buffers are setup 
        //assertKernelBuffersMade(kernelname);
        if(!kernelArgStatus.get(kernelname)){
            System.err.println("CLKernelHelper | Kernel Arguments "+kernelname+" not previously set. Will Attempt to set arguments now. ");
            setKernelArg(kernelname);
        }
        setQueueBuff(kernelname);
        queue.put2DRangeKernel(kernels.get(kernelname), 0, 0, gsizex, gsizey, lsizex, lsizey);
        queueOutputBuffers(kernelname);
        queue.finish();
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
        // Assert kernel buffers are setup 
        //assertKernelBuffersMade(kernelname);
        if(!kernelArgStatus.get(kernelname)){
            System.err.println("CLHelper | Kernel Arguments "+kernelname+"  not previously set. Will Attempt to set arguments now. ");
            setKernelArg(kernelname);
        }
        setQueueBuff(kernelname);
        queue.put3DRangeKernel(kernels.get(kernelname), 0, 0, 0,gsizex, gsizey, gsizez, lsizex, lsizey, lsizez);
        queueOutputBuffers(kernelname);
        queue.finish();
    }
    
    
    /**
    *       queueOutputBuffers asserts that output buffers get updated each time 
    *   the kernel is run.
    * 
    * @param kernelname - kernel that needs output buffers queued
    */
    private void queueOutputBuffers(String kernelname){
        ArrayList<Boolean> outflags;
        if(arghandler.getIntBuffers().get(kernelname) != null){
            outflags = arghandler.getIntBuffersIOFlags().get(kernelname);     
            for(int i=0;i<outflags.size();i++){
                if(outflags.get(i)){
                    arghandler.manageBuffer(kernelname,"int", i);
                }
            }
        }  
        if(arghandler.getFlBuffers().get(kernelname) != null){
            outflags = arghandler.getFlBuffersIOFlags().get(kernelname);     
            for(int i=0;i<outflags.size();i++){
                if(outflags.get(i)){
                    arghandler.manageBuffer(kernelname,"float", i);
                }
            }
        }
        if(arghandler.getLongBuffers().get(kernelname) != null){
            outflags = arghandler.getLongBuffersIOFlags().get(kernelname);     
            for(int i=0;i<outflags.size();i++){
                if(outflags.get(i)){
                    arghandler.manageBuffer(kernelname,"long", i);
                }
            }
        }
    }

    
    /**
    *       setQueueBuff sets the buffers in the OpenCL queue so the device knows  
    *   which buffers to access.
    * 
    * @param kernelname - kernel which needs buffers in queue
    */
    public void setQueueBuff(String kernelname){
        // get list of buffers for this kernel
        ArrayList<String> kernelTypes = argTypes.get(kernelname);
        int floatBuffInd = 0;
        int intBuffInd = 0;
        int longBuffInd = 0;
        String typecurr;
        // Set as buffers for the queue
        for(int j =0;j<kernelTypes.size();j++){
            typecurr = kernelTypes.get(j);
            if(typecurr.contains("int")&& typecurr.contains("buffer")){
                ArrayList<CLBuffer<IntBuffer>> outBuffers = arghandler.getIntBuffers().get(kernelname);
                CLBuffer<IntBuffer> push = outBuffers.get(intBuffInd);
                queue.putWriteBuffer(push,false);
                intBuffInd++;
            }else if(typecurr.contains("float") && typecurr.contains("buffer")){
                //outputClassString("Queuing Float Buffer for kernel : "+kernelname +"   of argn: "+floatBuffInd);
                ArrayList<CLBuffer<FloatBuffer>> outBuffers = arghandler.getFlBuffers().get(kernelname);
                CLBuffer<FloatBuffer> push = outBuffers.get(floatBuffInd);
                if(push.getClass().getSimpleName().contains("CLGLBuffer")){
                    queue.putAcquireGLObject((CLGLBuffer<?>)push);
                }else{
                    queue.putWriteBuffer(push,false);
                }
                floatBuffInd++;
            }else if(typecurr.contains("long") && typecurr.contains("buffer")){
                //outputClassString("Queuing Float Buffer");
                ArrayList<CLBuffer<LongBuffer>> outBuffers = arghandler.getLongBuffers().get(kernelname);
                CLBuffer<LongBuffer> push = outBuffers.get(longBuffInd);
                queue.putWriteBuffer(push,false);
                longBuffInd++;
            }
        }
    }
   
    
    /**
    *       getKernelIOTypes reads the given kernels source and parses out a
    *   list of arguments to use for type checking. Needs kernel with __kernel in source.
    * 
    * @param kernelname - kernel to check arguments
    * @param source - source of kernel
    * @return array of arguments required
    */
    private ArrayList<String> getKernelIOTypes(String kernelname, String source){
        //parse io types
        // kernel function starts with __kernel keyword
        source = source.substring(source.indexOf("__kernel"),source.length());
        String[] src=source.split("\\(");
        src = src[1].split("\\)");
        
        ArrayList<String> types = new ArrayList<String>();
        String[] typeSet = src[0].split(",");
        String proc,type="";
        boolean initFlBuff=false;
        boolean initIntBuff=false;
        boolean initLongBuff=false;
        boolean initFlArg=false;
        boolean initIntArg=false;
        boolean initLongArg=false;
        
        // Parse kernel IO arguments into buffers that will be needed
        for(int i=0;i<typeSet.length;i++){
            proc = typeSet[i];
            type="";
            
            if(proc.contains("global")){
                type = "global "+type;
            }else if(proc.contains("local")){
                type = "local "+type;
            } 

            if(proc.contains("const")){
                type = "const "+type;
            }
            
            if(proc.contains("float")){
               type = type+"float";
            }else if(proc.contains("int")){
                type = type+"int";
           }else if(proc.contains("long")){
                type = type+"long";
            } 
            
            if(proc.contains("*")){
                type = type+" buffer";
            }
            
            //outputClassString("Adding Argument Type: "+type);
            types.add(i, type);

            // initialize device buffers and arguments in ideal way
            if(type.contains("buffer")&&type.contains("float")&& !initFlBuff){
                ArrayList<CLBuffer<FloatBuffer>> push = new  ArrayList<CLBuffer<FloatBuffer>>();
                ArrayList<Boolean> out = new  ArrayList<Boolean>();
                arghandler.getFlBuffers().put(kernelname, push);
                arghandler.getFlBuffersIOFlags().put(kernelname, out);
            }else if(type.contains("buffer")&&type.contains("int")&& !initIntBuff){
                ArrayList<CLBuffer<IntBuffer>> push = new  ArrayList<CLBuffer<IntBuffer>>();
                ArrayList<Boolean> out = new  ArrayList<Boolean>();
               
                arghandler.getIntBuffers().put(kernelname, push);
                arghandler.getIntBuffersIOFlags().put(kernelname, out);
         }else if(type.contains("buffer")&&type.contains("long")&& !initIntBuff){
                ArrayList<CLBuffer<LongBuffer>> push = new  ArrayList<CLBuffer<LongBuffer>>();
                ArrayList<Boolean> out = new  ArrayList<Boolean>();
                arghandler.getLongBuffers().put(kernelname, push);
                arghandler.getLongBuffersIOFlags().put(kernelname, out);
     
            }else if(!type.contains("buffer")&&type.contains("int")&& !initIntArg){
                ArrayList<Integer> push = new  ArrayList<Integer>();
                arghandler.getIntArgs().put(kernelname, push);
            }else if(!type.contains("buffer")&&type.contains("long")&& !initIntArg){
                ArrayList<Long> push = new  ArrayList<Long>();
                arghandler.getLongArgs().put(kernelname, push);
            }else if(!type.contains("buffer")&&type.contains("float")&& !initFlBuff){
                ArrayList<Float> push = new  ArrayList<Float>();
                arghandler.getFloatArgs().put(kernelname, push);
            }
        }
        return types;
    }
    
    /**
    *       assertKernelBuffersMade checks if all required buffers have been made.
    * 
    * @param kernelname - kernel to check buffers 
    * @return true if buffers are all present
    */
    private boolean assertKernelBuffersMade(String kernelname){
        boolean made = false;
        int total = argTypes.get(kernelname).size();
        int count = 0;
        
        // count the amount of arguments
        if(arghandler.getIntBuffers().get(kernelname) != null){count += arghandler.getIntBuffers().get(kernelname).size();}
        if(arghandler.getFlBuffers().get(kernelname) != null){count += arghandler.getFlBuffers().get(kernelname).size();}
        if(arghandler.getLongBuffers().get(kernelname) != null){count += arghandler.getLongBuffers().get(kernelname).size();}
        if(arghandler.getFloatArgs().get(kernelname) != null){count += arghandler.getFloatArgs().get(kernelname).size();}
        if(arghandler.getIntArgs().get(kernelname) != null){count += arghandler.getIntArgs().get(kernelname).size();}
        if(arghandler.getLongArgs().get(kernelname) != null){count += arghandler.getLongArgs().get(kernelname).size();}

        if(count == total){made=true;}else{
        System.err.println("CLHelper | ERROR: ALL KERNELS Argument HAVE NOT BEEN INITIALIZED. KERNEL: "+kernelname
                +"----| kernel number of arguments: "+total+"    arguments initialized: "+count);
                if(arghandler.getIntBuffers().get(kernelname) != null){
                    System.err.print("Int Buffers: "+arghandler.getIntBuffers().get(kernelname).size());}
                if(arghandler.getFlBuffers().get(kernelname) != null){
                    System.err.print("   Float Buffers: "+arghandler.getFlBuffers().get(kernelname).size());}
                if(arghandler.getLongBuffers().get(kernelname) != null){
                    System.err.print("   Long Buffers: "+arghandler.getLongBuffers().get(kernelname).size());}
                if(arghandler.getFloatArgs().get(kernelname) != null){
                    System.err.print("    Float Args: "+arghandler.getFloatArgs().get(kernelname).size());}
                if(arghandler.getIntArgs().get(kernelname) != null){
                    System.err.print("     Int Args: "+arghandler.getIntArgs().get(kernelname).size());}
                if(arghandler.getLongArgs().get(kernelname) != null){
                    System.err.print("    Long Args: "+arghandler.getLongArgs().get(kernelname).size());}
        }
        return made;
    }
    
    
    /**
    *       setKernelArg sets the arguments of the given kernel by using 
    *   the arguments parsed from source.
    * 
    * @param kernelname - kernel to set arguments for
    */
    public void setKernelArg(String kernelname){
        if(!kernelArgStatus.get(kernelname)){
            setKernelArg(kernelname,false);
        }else{
            setKernelArg(kernelname,true);
        }
    }
    
    /**
    *       setKernelArg sets the arguments of the given kernel by using 
    *   the arguments parsed from source.
    * 
    * @param kernelname - kernel to set arguments for
    * @param setPrev - true if already set previously
    */
    public void setKernelArg(String kernelname, boolean setPrev){
        // if previously set override
        if(kernelArgStatus.get(kernelname)){
           setPrev = true; 
        }
        ArrayList<String> types = argTypes.get(kernelname);
        int intBuffInd = 0;
        int flBuffInd = 0;
        int longBuffInd = 0;
        int intInd = 0;
        int flInd = 0;
        int longInd = 0;
        // Index that is absolute with respect to types 
        int Ind = 0;
        // Get kernel and all arguments
        CLKernel kernel = kernels.get(kernelname);
        ArrayList<CLBuffer<FloatBuffer>> floatBuff = arghandler.getFlBuffers().get(kernelname);
        ArrayList<CLBuffer<IntBuffer>> intBuff = arghandler.getIntBuffers().get(kernelname);
        ArrayList<CLBuffer<LongBuffer>> longBuff = arghandler.getLongBuffers().get(kernelname);
        ArrayList<Float> floatArg = arghandler.getFloatArgs().get(kernelname);
        ArrayList<Integer> intArg = arghandler.getIntArgs().get(kernelname);
        ArrayList<Long> longArg = arghandler.getLongArgs().get(kernelname);
        
        // Set the arguments using the argument list
        for(int i=0;i < types.size();i++){
            String type = types.get(i);
            if(type.contains("int")&& type.contains("buffer") ){
                if(setPrev){
                    kernel.setArg(Ind,intBuff.get(intBuffInd));
                }else{
                    kernel.putArg(intBuff.get(intBuffInd));}
                intBuffInd++;
            }else if(type.contains("float")&& type.contains("buffer")){
                if(setPrev){
                    kernel.setArg(Ind,floatBuff.get(flBuffInd));
                }else{
                    kernel.putArg(floatBuff.get(flBuffInd));}                
                flBuffInd++;
            }else if(type.contains("long")&& type.contains("buffer")){
                if(setPrev){
                    kernel.setArg(Ind,longBuff.get(longBuffInd));
                }else{
                    kernel.putArg(floatBuff.get(longBuffInd));}                
                longBuffInd++;
            }else if(type.contains("float")&& !(type.contains("buffer"))){
                if(setPrev){
                    kernel.setArg(Ind,floatArg.get(flInd));                    
                }else{         
                    kernel.putArg(floatArg.get(flInd));
                }
                flInd++;
            }else if(type.contains("int")&& !(type.contains("buffer"))){
                if(setPrev){
                    kernel.setArg(Ind,intArg.get(intInd));                    
                }else{
                    kernel.putArg(intArg.get(intInd));
                }                
                intInd++;    
            }else if(type.contains("long")&& !(type.contains("buffer"))){
                if(setPrev){
                    kernel.setArg(Ind,longArg.get(longInd));                    
                }else{
                    kernel.putArg(longArg.get(longInd));
                }                
                longInd++;    
            }else{
                System.err.println("CLHelper | ERROR: UNCATEGORIZED ARGUMENT > type: "+type+"   argument number : "+i);}
                Ind++;
        }
        kernelArgStatus.put(kernelname,true);
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
        float mbused=0;
        
        ArrayList<CLBuffer<FloatBuffer>> kbuff = arghandler.getFlBuffers().get(kernelname);
        ArrayList<CLBuffer<IntBuffer>> ibuff = arghandler.getIntBuffers().get(kernelname);

        // integer buffers b used
        if(ibuff!=null){
        for(int i=0;i<ibuff.size();i++){
            mbused = ibuff.get(i).getCLSize() +mbused;
        }}

        // float buffers b used
        if(kbuff!=null){
        for(int i=0;i<kbuff.size();i++){
            mbused = kbuff.get(i).getCLSize() +mbused;
        }}
        
        // make into mb
        mbused = mbused/1000000;
        
        return (int) mbused;
    }
    
    /**
    *      setOutputMode sets level of output by turning off/on some output strings.
    * 
    * @param md - true if outputting 
    */
    public void setOutputMode(boolean md){    
        outputMode = md;
    }
    
    private void outputClassString(String msg){
        if(outputMode){System.out.println("CLKernelHelper | "+msg);}
    }    
    private void outputClassStarLine(){
        outputClassString("***********************************************************************");
    }    
    private void outputClassDashLine(){
        outputClassString("-------------------------------------------------------------------------------");
    }

    /**
    *       getKernelArgStatus returns true if kernel arguments have been 
    *   previously set for given kernel.
    * 
    * @param kernel - name of kernel to check status
    * @return true if kernel arguments have been set
    */
    public boolean getKernelArgStatus(String kernel){
        return kernelArgStatus.get(kernel);
    }
    
    /**
    *       readFile converts a file into a string. Useful for the building of 
    *   OpenCL kernel.
    * 
    * @param filename - filename of file to convert.
    * @return String version of file
    */
    private String readFile(String filename) {
        File f = new File(filename);
        try {
            byte[] bytes = Files.readAllBytes(f.toPath());
            return new String(bytes,"UTF-8");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return "";
    }

}
