package scikit.opencl;

/**
* 
*    @(#)   CLArgHelper
*/ 

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import static com.jogamp.opencl.CLMemory.Mem.READ_ONLY;
import static com.jogamp.opencl.CLMemory.Mem.READ_WRITE;
import static com.jogamp.opencl.CLMemory.Mem.WRITE_ONLY;
import com.jogamp.opencl.gl.CLGLBuffer;
import com.jogamp.opencl.gl.CLGLContext;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.HashMap;


/**
* 
*    CLArgHelper handles the management of buffers created for kernels.
* 
* <br>
* 
* @author      James B. Silva <jbsilva @ bu.edu>                 
* @since       2013
*/
public class CLArgHelper{
    private CLContext context;
    private CLGLContext contextGLCL;
    private boolean glEnabled = false;
    private CLCommandQueue queue;
    private boolean outputMsg = false;
    private boolean printJustVals = true;
    private double lastBufferMaxValue = 0;
    private HashMap<String,ArrayList<CLBuffer<FloatBuffer>>> flBuffers;
    private HashMap<String,ArrayList<Boolean>> flBuffersIOFlags;
    private HashMap<String,ArrayList<CLBuffer<LongBuffer>>> longBuffers;
    private HashMap<String,ArrayList<Boolean>> longBuffersIOFlags;
    private HashMap<String,ArrayList<CLBuffer<IntBuffer>>> intBuffers;
    private HashMap<String,ArrayList<Boolean>> intBuffersIOFlags;
    private HashMap<String,ArrayList<Integer>> intArgs;
    private HashMap<String,ArrayList<Float>> floatArgs;    
    private HashMap<String,ArrayList<Long>> longArgs;
    
    public HashMap<String,ArrayList<CLBuffer<FloatBuffer>>> getFlBuffers(){return flBuffers;}
    public HashMap<String,ArrayList<Boolean>> getFlBuffersIOFlags(){return flBuffersIOFlags;}
    public HashMap<String,ArrayList<CLBuffer<LongBuffer>>> getLongBuffers(){return longBuffers;}
    public HashMap<String,ArrayList<Boolean>> getLongBuffersIOFlags(){return longBuffersIOFlags;}
    public HashMap<String,ArrayList<CLBuffer<IntBuffer>>> getIntBuffers(){return intBuffers;}
    public HashMap<String,ArrayList<Boolean>> getIntBuffersIOFlags(){return intBuffersIOFlags;}
    public HashMap<String,ArrayList<Integer>> getIntArgs(){return intArgs;}
    public HashMap<String,ArrayList<Float>> getFloatArgs(){return floatArgs;}    
    public HashMap<String,ArrayList<Long>> getLongArgs(){return longArgs;}
    
    public CLArgHelper(CLHelper clhandle){
        initializeOpenCLArgs(clhandle);
    }  
    
    /**
    *   initializeOpenCLArgs setups the containers needed for handling arguments.
    * 
    * @param clhandle - the OpenCLHandler class containing the context and queue to manage. 
    */
    public void initializeOpenCLArgs(CLHelper clhandle){
        // Initialize all list and maps
        flBuffers = new HashMap<String,ArrayList<CLBuffer<FloatBuffer>>> ();
        intBuffers = new HashMap<String,ArrayList<CLBuffer<IntBuffer>>>();
        longBuffers = new HashMap<String,ArrayList<CLBuffer<LongBuffer>>>();
        flBuffersIOFlags = new HashMap<String,ArrayList<Boolean>>();
        intBuffersIOFlags = new HashMap<String,ArrayList<Boolean>>();
        longBuffersIOFlags = new HashMap<String,ArrayList<Boolean>>();
        
        intArgs = new HashMap<String,ArrayList<Integer>>();
        floatArgs  = new HashMap<String,ArrayList<Float>>();
        longArgs = new HashMap<String,ArrayList<Long>>();
        
        context = clhandle.getContext();
        contextGLCL = clhandle.getContextGLCL();
        if(contextGLCL == null){
            glEnabled = false;
        }else{glEnabled = true;}
        queue = clhandle.getQueue();  
    }
    
    /**
    *       manageBuffer - this variant just gets buffers but doesnt display data. Useful
    *   for the main handler class to queue output buffers to read to update.
    * 
    * 
    * @param kernelname - kernel to add buffer for
    * @param type - type of buffer - int , float, or long
    * @param argn - argument type entry number
    * @return created buffer
    */
    public CLBuffer manageBuffer(String kernelname, String type, int argn){
        return manageBuffer(kernelname,  type, argn,  "get",  0, "",  0 ,  null, null,false,0);
    }

//            return manageBuffer(kernelname,  type, argn,  updateMode,  size, readwrite,  fill ,  s0,fillMod, inbuff);

    /**
    *       manageBuffer variant that sets a kernel argument to the given buffer. 
    * 
    * 
    * @param kernelname - kernel to add buffer for
    * @param type - buffer type to set ie int , float , or long.
    * @param argn - argument type entry number
    * @param  inbuff - buffer to set argument to
    * @return created buffer
    */
    public CLBuffer manageBuffer(String kernelname, String type, int argn,  CLBuffer inbuff){
        if(checkBufferCompat(type,inbuff)){
            return manageBuffer(kernelname,  type, argn,  "set",  0, "",  0 ,  null, inbuff,false,0);}
        else{
            System.err.println("CLArgHelper | ATTEMPTING TO SET BUFFER OF TYPE "+
                    inbuff.getBuffer().getClass().getSimpleName()+" TO AN EXISTING BUFFER OF TYPE "+type);
            return null;
        }
    }
    
    /**
    *       checkBufferCompat checks that the buffer is of the desired type.
    * 
    * @param type - type to compare (int , float, long)
    * @param inbuff - buffer to determine type
    * @return true if compatible
    * @return created buffer
    */
    private boolean checkBufferCompat(String type, CLBuffer inbuff){
        boolean bufferGood=false;
        if(type.equalsIgnoreCase("int")){
            if(inbuff.getBuffer().getClass().getSimpleName().contains("int") ||
               inbuff.getBuffer().getClass().getSimpleName().contains("Int")){
            bufferGood=true;
            }
        }else if(type.equalsIgnoreCase("float")){
            if(glEnabled || inbuff.getClass().getSimpleName().contains("CLGLBuffer") ){
                bufferGood=true;
            }
            if(!bufferGood){
                if(inbuff.getBuffer().getClass().getSimpleName().contains("Float") ||
                   inbuff.getBuffer().getClass().getSimpleName().contains("float")){
                bufferGood=true;
                }
            }
        }else if(type.equalsIgnoreCase("long")){
            if(inbuff.getBuffer().getClass().getSimpleName().contains("long") ||
               inbuff.getBuffer().getClass().getSimpleName().contains("Long")){
            bufferGood=true;
            }
        }
        inbuff.getBuffer().rewind();
        return bufferGood;
    } 
    
    
    /**
    *       manageBuffer variant that creates a buffer but does not fill the buffer. 
    * 
    * @param kernelname - kernel to add buffer for
    * @param type - type for argument (int, long, or float)
    * @param argn - argument type entry number
    * @param size - size of buffer
    * @param readwrite - "r" for read , "w" for write, else default read/write. 
    * @return created buffer
    */
    public CLBuffer manageBuffer(String kernelname, String type, int argn, int size,String readwrite){
        return manageBuffer(kernelname,  type, argn,  "create",  size, readwrite,  0 , null, null,false,0);
    } 

    
    /**
    *       manageBuffer variant that creates and fills an integer buffer then adds it to 
    *   the list of  buffers for this kernel at given argument entry number. 
    * 
    * @param kernelname - kernel to add buffer for
    * @param s0 - initial value of all buffer entries
    * @param size - size of buffer
    * @param readwrite - "r" for read , "w" for write, else default read/write.
    * @param argn - argument type entry number
    * @return created buffer
    */
    public CLBuffer manageBuffer(String kernelname, int s0, int size,String readwrite, int argn){
        ArrayList<Integer> push = new ArrayList<Integer>();
        push.add(s0);
        return manageBuffer(kernelname,  "int", argn,  "create",  size, readwrite,  1 ,push , null,false,0);
    }
    
    /**
    *       manageBuffer variant that creates and fills a float buffer then adds it to 
    *   the list of float buffers for this kernel at given argument entry number. 
    * 
    * @param kernelname - kernel to add buffer for
    * @param s0 - initial value of all buffer entries
    * @param size - size of buffer
    * @param readwrite - "r" for read , "w" for write, else default read/write.
    * @param argn - argument type entry number
    * @return created buffer
    */
    public CLBuffer manageBuffer(String kernelname, float s0, int size,String readwrite, int argn){
        ArrayList<Float> push = new ArrayList<Float>();
        push.add(s0);
        return manageBuffer(kernelname,  "float", argn,  "create",  size, readwrite,  1 ,push , null,false,0);
    }
    
    
    /**
    *       quickGetLastBufferMax gets the maximum value calculated from the last
    *   retrieved buffer
    * 
    *   @return maximum value of last quesried buffer
    */
    public double quickGetLastBufferMax(){
        return lastBufferMaxValue;
    }
   
    /**
    *       manageBuffer variant that creates and fills a long buffer then adds it to 
    *   the list of long buffers for this kernel at given argument entry number. 
    * 
    * @param kernelname - kernel to add buffer for
    * @param s0 - initial value of all buffer entries
    * @param size - size of buffer
    * @param readwrite - "r" for read , "w" for write, else default read/write.
    * @param argn - argument type entry number
    * @return created buffer
    */
    public CLBuffer manageBuffer(String kernelname, long s0, int size,String readwrite, int argn){
        ArrayList<Long> push = new ArrayList<Long>();
        push.add(s0);
        return manageBuffer(kernelname,  "long", argn,  "create",  size, readwrite,  1 ,push , null,false,0);
    }
    
     /**
    *       manageBuffer variant that creates and fills a buffer with an array list then adds it to 
    *   the list of buffers for this kernel at given argument entry number. 
    * 
    * 
    * @param kernelname - kernel to add buffer for
    * @param type - type for argument (int, long, or float)
    * @param argn - argument type entry number
    * @param s0 - initial array of value of all buffer entries
    * @param readwrite - "r" for read , "w" for write, else default read/write.
    * @return created buffer
    */
    public CLBuffer manageBuffer(String kernelname, String type,int argn,ArrayList s0,String readwrite){
        return manageBuffer(kernelname, type, argn,  "create",  s0.size(), readwrite,  1 ,s0 , null,false,0);
    }
    
    /**
    *       manageBuffer variant that creates and fills a buffer with elements of an 
    *   array list with the option of doing so with repetitions of this array then adds it to 
    *   the list of buffers for this kernel at given argument entry number. 
    * 
    * 
    * @param kernelname - kernel to add buffer for
    * @param type - type for argument (int, long, or float)
    * @param argn - argument type entry number
    * @param size - size of buffer
    * @param s0 - initial value array of all buffer entries
    * @param readwrite - "r" for read , "w" for write, else default read/write.
    * @param modFill - true if filling buffer with repetitions of array list values
    * @return created buffer
    */
    public CLBuffer manageBuffer(String kernelname, String type,int argn,ArrayList s0,int size,
            String readwrite, boolean modFill){
        if(modFill){
             return manageBuffer(kernelname, type, argn,  "create",  size, readwrite,  2 ,s0 , null,false,0);}
        else{return manageBuffer(kernelname, type, argn,  "create",  size, readwrite,  1 ,s0 , null,false,0);}
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
    * @param fill - 1 if regular filling buffer, 2 or greater filling buffer with repetitions of array list values
    * @param s0 - initial value array of all buffer entries
    * @param  inbuff - buffer to set argument to
    * @return created buffer
    */
    public CLBuffer manageBuffer(String kernelname, String type, int argn, String updateMode, int size,
            String readwrite, int fill , ArrayList s0,  CLBuffer inbuff, boolean glShared, int glInd){
       
        boolean fillMod = getFillMod(fill);
        
        boolean invalidType = true;
        CLBuffer buff;
        
        // Create clbuffer object
        if(updateMode.equalsIgnoreCase("create") || updateMode.equalsIgnoreCase("creates")){
            if(glShared){
                System.out.println("Adding to context : glInd: "+ glInd+"   of size: "+size+"   arg: "+argn);
                if(type.contains("oat")){
                    buff = createSharedFlBufferGL(s0,fillMod , size, readwrite, glInd);
                }else{
                    buff = null;
                }
            }else{
                buff = createBuffer(type,size,readwrite);
                // fill the buffer if required
                if(fill>0){
                    buff = fillBuffer(buff,type,s0,fillMod);
                }
            }
            if(readwrite.equalsIgnoreCase("r")){
                pushBuffer(kernelname,argn,buff,type,false,false);
            }else{
                pushBuffer(kernelname,argn,buff,type,true,false);
            }
        }else if(updateMode.equalsIgnoreCase("get")){
            buff = getBuffer(kernelname,type,argn);
        }else if(updateMode.equalsIgnoreCase("set")){
            // fill the buffer if set on an array list
            if(fill>0){
                buff = getBuffer(kernelname,type,argn);
                buff = fillBuffer(buff,type,s0,fillMod);
            }else{
                buff = setBuffer(kernelname,type,argn,inbuff);
            }
        }else{
            buff = null; System.err.println("CLArgHelper | INVALID MANAGE BUFFER MODE");
        }
        
        return buff; 
    }
    
    /**
    *       getFillMod converts int fill code to a boolean that determines fill 
    *   using mod filling 
    * @param fill - int fill code
    * @return true if fill with mod
    */
    private boolean getFillMod(int fill){
        if(fill>1){return true;}else{return false;}
    }
    
    /**
    *       setBuffer sets a buffer to the given CLBuffer  
    * 
    * @param kernelname - kernel to set buffer for  
    * @param type - type for argument (int, long, or float)
    * @param argn - argument type entry number
    * @param  buffer - buffer to set argument to
    * @return created buffer
    */
    public CLBuffer setBuffer(String kernelname, String type,int argn,CLBuffer buffer){
        boolean output;
        if(type.equalsIgnoreCase("int")){
            output = intBuffersIOFlags.get(kernelname).get(argn);
            pushBuffer(kernelname,argn,buffer,type,output,false);
        }else if(type.equalsIgnoreCase("float")){
            output = flBuffersIOFlags.get(kernelname).get(argn);
            pushBuffer(kernelname,argn,buffer,type,output,false);
        }else if(type.equalsIgnoreCase("long")){
            output = longBuffersIOFlags.get(kernelname).get(argn);
            pushBuffer(kernelname,argn,buffer,type,output,false);     
        }     
        return buffer;
    }
    
    /**
    *       getBuffer gets the buffer from device after updating.
    * 
    * @param kernelname - kernel to set buffer for  
    * @param type - type for argument (int, long, or float)
    * @param argn - argument type entry number
    * @return created buffer
    */
    public CLBuffer getBuffer(String kernelname, String type,int argn){
        CLBuffer buff;
        if(type.equalsIgnoreCase("int")){
            CLBuffer<IntBuffer> buffer=intBuffers.get(kernelname).get(argn);
            queue.putReadBuffer(buffer, true);
            buff = buffer;
        }else if(type.equalsIgnoreCase("float")){
            CLBuffer<FloatBuffer> buffer=flBuffers.get(kernelname).get(argn);
            if(buffer.getClass().getSimpleName().contains("CLGLBuffer")){
                queue.putReleaseGLObject((CLGLBuffer<?>)buffer);
            }else{
                queue.putReadBuffer(buffer, true);        
            }
            buff = buffer;
        }else if(type.equalsIgnoreCase("long")){
            CLBuffer<LongBuffer> buffer=longBuffers.get(kernelname).get(argn);
            queue.putReadBuffer(buffer, true);
            buff = buffer;
        }else{
            System.err.println("CLArgHelper | GETTING INCOMPATIBLE BUFFER TYPE");buff=null;
        }
        return buff;
    }
    
    /**
    *      createBuffer creates a buffer on the device.
    * 
    * @param type - type for argument (int, long, or float)
    * @param size - size of buffer
    * @param readwrite - "r" for read , "w" for write, else default read/write.
    * @return created buffer
    */
    public CLBuffer createBuffer(String type, int size, String readwrite){
        CLBuffer buffer;
        if(type.equalsIgnoreCase("int")){
            if(readwrite.equalsIgnoreCase("w")){
                if(glEnabled){
                    buffer = contextGLCL.createIntBuffer(size, WRITE_ONLY);
                }else{
                    buffer = context.createIntBuffer(size, WRITE_ONLY);
                }
            }else if(readwrite.equalsIgnoreCase("r")){
                if(glEnabled){
                    buffer = contextGLCL.createIntBuffer(size, READ_ONLY);
                }else{
                    buffer = context.createIntBuffer(size, READ_ONLY);
                }
            }else{
                if(glEnabled){
                    buffer = contextGLCL.createIntBuffer(size, READ_WRITE);
                }else{
                    buffer = context.createIntBuffer(size, READ_WRITE);
                }
            }    
        }else if(type.equalsIgnoreCase("float") || type.equalsIgnoreCase("fl")){
            if(readwrite.equalsIgnoreCase("w")){
                if(glEnabled){
                    buffer = contextGLCL.createFloatBuffer(size, WRITE_ONLY);
                }else{
                    buffer = context.createFloatBuffer(size, WRITE_ONLY);
                }
            }else if(readwrite.equalsIgnoreCase("r")){
                if(glEnabled){
                    buffer = contextGLCL.createFloatBuffer(size, READ_ONLY);
                }else{
                    buffer = context.createFloatBuffer(size, READ_ONLY);
                }
            }else{
                if(glEnabled){
                    buffer = contextGLCL.createFloatBuffer(size, READ_WRITE);
                }else{
                    buffer = context.createFloatBuffer(size, READ_WRITE);
                }
            }
        }else if(type.equalsIgnoreCase("long")){
            if(readwrite.equalsIgnoreCase("w")){
                if(glEnabled){
                    buffer = contextGLCL.createLongBuffer(size, WRITE_ONLY);
                }else{
                    buffer = context.createLongBuffer(size, WRITE_ONLY);
                }
            }else if(readwrite.equalsIgnoreCase("r")){
                if(glEnabled){
                    buffer = contextGLCL.createLongBuffer(size, READ_ONLY);
                }else{
                    buffer = context.createLongBuffer(size, READ_ONLY);
                }
            }else{
                if(glEnabled){
                    buffer = contextGLCL.createLongBuffer(size, READ_WRITE);
                }else{
                    buffer = context.createLongBuffer(size, READ_WRITE);
                }
            }
        }else if(type.equalsIgnoreCase("image2d")){
            System.err.println("CLArgHelper | IMAGE NOT IMPLEMENTED YET.");buffer=null;
        }else{
            buffer = null; 
            System.err.println("CLArgHelper | BUFFER TYPE IS NOT PROPERLY DEFINED. USE LONG,INT,FLOAT,IMAGE2D.");
        }
    
        return buffer;
    }
    
    /**
    *      createSharedFlBufferGL creates a buffer on the device.
    * 
    * @param size - size of buffer
    * @param readwrite - "r" for read , "w" for write, else default read/write.
    * @return created buffer
    */
    public CLGLBuffer<?> createSharedFlBufferGL(ArrayList data,boolean fillmod, int size, String readwrite,int glIndex){
        CLGLBuffer<?> buffer;
        float[] temp = makeFlArrFromArrList(data, size,fillmod);
        FloatBuffer inbuff = makeFloatBuffer(temp);
        if(readwrite.equalsIgnoreCase("w")){
            buffer = contextGLCL.createFromGLBuffer(inbuff,glIndex, size, CLGLBuffer.Mem.WRITE_ONLY);
        }else if(readwrite.equalsIgnoreCase("r")){
            buffer = contextGLCL.createFromGLBuffer(inbuff,glIndex, size, CLGLBuffer.Mem.READ_ONLY);
        }else{
            buffer = contextGLCL.createFromGLBuffer(inbuff,glIndex, size, CLGLBuffer.Mem.READ_WRITE);
        }
        System.err.println("CLArgHelper |  Shared float buffer created | "+ buffer.getBuffer()); 
        
        return buffer;
    }
    
    // helper function for converting to useful float array
    private float[] makeFlArrFromArrList(ArrayList al, int size, boolean fillmod){
        float[] temp = new float[size];
        for(int u = 0;u< size; u++){
            if(fillmod){
                temp[u] = (float) al.get(u%al.size());
            }else{
                if(u<al.size()){
                    temp[u] = (float) al.get(u);
                }
            }
        }
        return temp;
    }
    /**
    * Make a direct NIO FloatBuffer from an array of floats
    * @param arr The array
    * @return The newly created FloatBuffer
    */
    public FloatBuffer makeFloatBuffer(float[] arr) {
        // float is 4 bytes
        ByteBuffer bb = ByteBuffer.allocateDirect(arr.length*4);
        bb.order(ByteOrder.nativeOrder());
        FloatBuffer fb = bb.asFloatBuffer();
        fb.put(arr);
        fb.position(0);
        return fb;
    }
    
    /**
    *      checkFillProperType checks if the array list is of type compatible with 
    *  the buffer.
    * 
    * @param clbuff - buffer of interest
    * @param arr - array to check type
    * @return true if filling with right type
    */
    @SuppressWarnings("unchecked")
    public boolean checkFillProperType(CLBuffer clbuff, ArrayList arr){
        boolean good=false;
      
        if(clbuff.getBuffer().getClass().getSimpleName().contains("int") ||
            clbuff.getBuffer().getClass().getSimpleName().contains("Int")){
            arr.add(0);
            if(arr.get(0).getClass().getSimpleName().equalsIgnoreCase("Integer")){
                good=true;
            }
            arr.remove(arr.size()-1);
        }else if(clbuff.getBuffer().getClass().getSimpleName().contains("Float") ||
            clbuff.getBuffer().getClass().getSimpleName().contains("float")){
            arr.add(0.0f);
            if(arr.get(0).getClass().getSimpleName().equalsIgnoreCase("Float")){
                good=true;
            }
            arr.remove(arr.size()-1);                
        }else if(clbuff.getBuffer().getClass().getSimpleName().contains("long") ||
            clbuff.getBuffer().getClass().getSimpleName().contains("Long")){
            arr.add(0l);
            if(arr.get(0).getClass().getSimpleName().equalsIgnoreCase("Long")){
                good=true;
            }
            arr.remove(arr.size()-1);
        }

        //assert buffer is rewound
        clbuff.getBuffer().rewind();    
        return good;
    
    }
    
    /**
    *      fillBuffer fills a buffer with the given array.
    * 
    * @param buffer
    * @param type - type for argument (int, long, or float)
    * @param s0 - initial value array of all buffer entries
    * @param fillMod - true if filling buffer with repetitions of array list values
    * @return filled buffer 
    */
    @SuppressWarnings("unchecked")
    public CLBuffer fillBuffer(CLBuffer buffer,String type, ArrayList s0, boolean fillMod){    
        if(!checkFillProperType(buffer, s0)){
            System.err.println("CLArgHelper | ATTEMPTING TO FILL BUFFER OF TYPE "+
                    buffer.getBuffer().getClass().getSimpleName()
                    +" WITH INCOMPATIBLE TYPE ARRAY LIST OR ARRAY OF TYPE :"+type);
        }
        
        if(type.equalsIgnoreCase("int")){
            IntBuffer buff = (IntBuffer) buffer.getBuffer();
            int i=0;
            int size = s0.size();
            if(fillMod){
                while(buff.remaining() != 0){buff.put((int)s0.get(i%size));i++;}
            }else if(!fillMod && size != 1){
                while(buff.remaining() != 0){buff.put((int)s0.get(i));i++;}    
            }else{
                while(buff.remaining() != 0){buff.put((int)s0.get(0));i++;}
            }
            buff.rewind();
            buffer.use(buff);
        }else if(type.equalsIgnoreCase("float")){
            FloatBuffer buff = (FloatBuffer) buffer.getBuffer();
            int i=0;
            int size = s0.size();
            if(fillMod){
                while(buff.remaining() != 0){buff.put((float)s0.get(i%size));i++;}
            }else if(!fillMod && size != 1){
                while(buff.remaining() != 0){buff.put((float)s0.get(i));i++;}    
            }else{
                while(buff.remaining() != 0){buff.put((float)s0.get(0));i++;}
            }
            buff.rewind();
            buffer.use(buff);
        }else if(type.equalsIgnoreCase("long")){
            LongBuffer buff = (LongBuffer) buffer.getBuffer();
            int i=0;
            int size = s0.size();
            if(fillMod){
                while(buff.remaining() != 0){buff.put((long)s0.get(i%size));i++;}
            }else if(!fillMod && size != 1){
                while(buff.remaining() != 0){buff.put((long)s0.get(i));i++;}    
            }else{
                while(buff.remaining() != 0){buff.put((long)s0.get(0));i++;}
            }
            buff.rewind();
            buffer.use(buff);
        }else{
            System.err.println("CLArgHelper | ATTEMPTING TO FILL A INCOMPATIBLE TYPE.");
        }

        return buffer;
    }
    
    
    /**
    *          pushBuffer adds a buffer to the list of buffers for this context.
    * 
    * @param kernelname - kernel to add buffer for
    * @param argn - argument type entry number
    * @param buffer - buffer to add to kernel arguments
    * @param type - type for argument (int, long, or float)
    * @param outputBuffer - true if buffer is output type ie a write or readwrite buffer
    * @param setMode - true if setting an existing argument instead of adding
    */
    @SuppressWarnings("unchecked")
    public void pushBuffer(String kernelname, int argn, CLBuffer buffer,String type, boolean outputBuffer, boolean setMode){        
        if(!checkBufferCompat(type, buffer)){
            System.err.println("CLArgHelper | ATTEMPTING TO PUSH BUFFER OF TYPE "+
                    buffer.getBuffer().getClass().getSimpleName()+" TO AN EXISTING BUFFERS OF TYPE "+type);
        };
        
        if(type.equalsIgnoreCase("int")){
            // Initialize buffer type array if not already
            ArrayList<CLBuffer<IntBuffer>> kbuff = intBuffers.get(kernelname);
            ArrayList<Boolean> obuff = intBuffersIOFlags.get(kernelname);
                
            if(kbuff == null){ 
                System.err.println("CLArgHelper | Initializing set of integer kernel buffers.");
                kbuff = new  ArrayList<CLBuffer<IntBuffer>>();
                obuff = new  ArrayList<Boolean>();
            }

            // push buffer into buffers list        
            if(setMode){
                kbuff.set(argn, ((CLBuffer<IntBuffer>) buffer));
            }else{
                kbuff.add(argn, ((CLBuffer<IntBuffer>) buffer));
            }
            
            intBuffers.put(kernelname, kbuff);
            if(outputBuffer){
                if(setMode){obuff.set(argn,true);}else{
                obuff.add(argn,true);}
            }else{
                if(setMode){obuff.set(argn,false);}else{
                obuff.add(argn,false);}
            }
            intBuffersIOFlags.put(kernelname, obuff);
        }else if(type.equalsIgnoreCase("float")){
            // Initialize buffer type array if not already
            ArrayList<CLBuffer<FloatBuffer>> kbuff = flBuffers.get(kernelname);
            ArrayList<Boolean> obuff = flBuffersIOFlags.get(kernelname);
    
            if(kbuff == null){ 
                System.err.println("CLArgHelper | Initializing set of float kernel buffers.");
                kbuff = new  ArrayList<CLBuffer<FloatBuffer>>();
                obuff = new  ArrayList<Boolean>();
            }
            // push buffer into buffers list        
            if(setMode){
                kbuff.set(argn, ((CLBuffer<FloatBuffer>) buffer));
            }else{
                kbuff.add(argn, ((CLBuffer<FloatBuffer>) buffer));
            }                    
            flBuffers.put(kernelname, kbuff);
            if(outputBuffer){
                if(setMode){obuff.set(argn,true);}else{obuff.add(argn,true);}
            }else{
                if(setMode){obuff.set(argn,false);}else{obuff.add(argn,false);}
            }
            flBuffersIOFlags.put(kernelname, obuff);
        }else if(type.equalsIgnoreCase("long")){
            // Initialize buffer type array if not already
            ArrayList<CLBuffer<LongBuffer>> kbuff = longBuffers.get(kernelname);
            ArrayList<Boolean> obuff = intBuffersIOFlags.get(kernelname);
            if(kbuff == null){ 
                System.err.println("CLArgHelper | Initializing set of long kernel buffers.");
                kbuff = new  ArrayList<CLBuffer<LongBuffer>>();
                obuff = new  ArrayList<Boolean>();
            }

            // push buffer into buffers list        
            if(setMode){
                kbuff.set(argn, ((CLBuffer<LongBuffer>) buffer));}else{
                kbuff.add(argn, ((CLBuffer<LongBuffer>) buffer));}

            longBuffers.put(kernelname, kbuff);
        
            if(outputBuffer){
                if(setMode){obuff.set(argn,true);}else{
                obuff.add(argn,true);}
            }else{
                if(setMode){obuff.set(argn,false);}else{
                obuff.add(argn,false);}
            }
            longBuffersIOFlags.put(kernelname, obuff);
        }
    }
    
    /**
    *       copyFLBufferAcrossKernel copies a buffer from source kernel float buffer list
    *   into destination kernel float buffer list.
    * 
    * @param type - type for argument (int, long, or float)
    * @param skernel - source kernel name
    * @param sargn - source kernel float argument number
    * @param dkernel - destination kernel name
    * @param dargn - destination kernel float argument number
    */
    public void copyBufferAcrossKernel(String type,String skernel,int sargn,String dkernel,int dargn){
        copyBufferAcrossKernel(type, skernel, sargn, dkernel, dargn, false);
    }

    /**
    *       copyFLBufferAcrossKernel copies a buffer from source kernel float buffer list
    *   into destination kernel float buffer list.
    * 
    * @param type - type for argument (int, long, or float)
    * @param skernel - source kernel name
    * @param sargn - source kernel float argument number
    * @param dkernel - destination kernel name
    * @param dargn - destination kernel float argument number
    * @param setMode - true if in set mode
    */
    public void copyBufferAcrossKernel(String type,String skernel,int sargn,String dkernel,int dargn, boolean setMode){
        // get source buffer
        if(type.equalsIgnoreCase("float")){
            CLBuffer<FloatBuffer> inBuff = flBuffers.get(skernel).get(sargn);
            boolean inBuffIO = flBuffersIOFlags.get(skernel).get(sargn);
            
            ArrayList<CLBuffer<FloatBuffer>> kbuff = flBuffers.get(dkernel);
            ArrayList<Boolean> kbuffIO = flBuffersIOFlags.get(dkernel);           
     
            if(kbuff==null || kbuff.size() == 0){
                kbuff = new ArrayList<CLBuffer<FloatBuffer>>();
                kbuffIO = new ArrayList<Boolean>();
            }
            
            if(kbuff.size() > dargn || setMode){
                kbuff.set(dargn, inBuff);
                kbuffIO.set(dargn, inBuffIO);
            }else{
                kbuff.add(dargn, inBuff);
                kbuffIO.add(dargn, inBuffIO);
            }
            // Update destination kernel with new buffer
            flBuffers.put(dkernel, kbuff);
            flBuffersIOFlags.put(dkernel, kbuffIO);
        }else if(type.equalsIgnoreCase("int")){
            CLBuffer<IntBuffer> inBuff = intBuffers.get(skernel).get(sargn);
            boolean inBuffIO = intBuffersIOFlags.get(skernel).get(sargn);
            ArrayList<CLBuffer<IntBuffer>> kbuff = intBuffers.get(dkernel);
            ArrayList<Boolean> kbuffIO = intBuffersIOFlags.get(dkernel);
            if(kbuff==null){
                kbuff = new ArrayList<CLBuffer<IntBuffer>> ();
                kbuffIO = new ArrayList<Boolean>();
            }
            if(kbuff.size()< dargn|| setMode){
                kbuff.set(dargn, inBuff);
                kbuffIO.set(dargn, inBuffIO);
            }else{kbuff.add(dargn, inBuff);
                kbuffIO.add(dargn, inBuffIO);
            }
            // Update destination kernel with new buffer
            intBuffers.put(dkernel, kbuff);
            intBuffersIOFlags.put(dkernel, kbuffIO);
        }else if(type.equalsIgnoreCase("long")){
            CLBuffer<LongBuffer> inBuff = longBuffers.get(skernel).get(sargn);
            boolean inBuffIO = longBuffersIOFlags.get(skernel).get(sargn);
            ArrayList<CLBuffer<LongBuffer>> kbuff = longBuffers.get(dkernel);
            ArrayList<Boolean> kbuffIO = longBuffersIOFlags.get(dkernel);
            if(kbuff==null){
                kbuff = new ArrayList<CLBuffer<LongBuffer>> ();
                kbuffIO = new ArrayList<Boolean>();
            }
            if(kbuff.size()< dargn|| setMode){
                kbuff.set(dargn, inBuff);
                kbuffIO.set(dargn, inBuffIO);
            }else{kbuff.add(dargn, inBuff);
                kbuffIO.add(dargn, inBuffIO);
            }
            // Update destination kernel with new buffer
            longBuffers.put(dkernel, kbuff);
            longBuffersIOFlags.put(dkernel, kbuffIO);   
        }
    }
    
    /**
    *       setIntArg creates an int and adds it to the arguments for kernel.
    * 
    * @param kernelname - name of kernel to add argument to list
    * @param argn - argument type entry number
    * @param val - initial value of int
    */
    public void setIntArg(String kernelname,int argn, int val){
        ArrayList<Integer> iargs = intArgs.get(kernelname);
        if(iargs==null){
            System.err.println("CLArgHelper | Initializing single int arg kernel.");
            iargs = new ArrayList<Integer>();
        } 
        if(iargs.size()> argn){
            iargs.set(argn,val);
        }else{
            iargs.add(argn,val);
        }
        intArgs.put(kernelname,iargs);
    }

    /**
    *       setFloatArg creates an float and adds it to the arguments for kernel.
    * 
    * @param kernelname - name of kernel to add argument to list
    * @param argn - argument type entry number
    * @param val - initial value of float
    */
    public void setFloatArg(String kernelname,int argn, float val){
        ArrayList<Float> fargs = floatArgs.get(kernelname);
        if(fargs==null){
            System.err.println("CLArgHelper | Initializing single float arg kernel.");
            fargs = new ArrayList<Float>();
        }
        
        if(fargs.size()> argn){
        fargs.set(argn,val);}else{
            fargs.add(argn,val);
        }
        floatArgs.put(kernelname,fargs);
    }
    
    /**
    *       setLongArg creates an Long and adds it to the arguments for kernel.
    * 
    * @param kernelname - name of kernel to add argument to list
    * @param argn - argument type entry number
    * @param val - initial value of Long
    */
    public void setLongArg(String kernelname,int argn, long val){
        ArrayList<Long> largs = longArgs.get(kernelname);
        if(largs==null){
            System.err.println("CLArgHelper | Initializing single kernel long arg.");
            largs = new ArrayList<Long>();
        }
        if(largs.size()> argn){
            largs.set(argn,val);}else{
            largs.add(argn,val);
        }
        longArgs.put(kernelname,largs);
    }

    
    /**
    *       getIntArg get an int from the kernel set.
    * 
    * @param kernelname - name of kernel to get int from
    * @param argn - argument type entry number
    */
    public int getIntArg(String kernelname,int argn){
        return intArgs.get(kernelname).get(argn);
    }
    
    /**
    *       getLongArg get a long from the kernel set.
    * 
    * @param kernelname - name of kernel to get long from
    * @param argn - argument type entry number
    */
    public long getLongArg(String kernelname,int argn){
        return longArgs.get(kernelname).get(argn);
    }
    
    /**
    *       getFloatArg get a float from the kernel set.
    * 
    * @param kernelname - name of kernel to get float from
    * @param argn - argument type entry number
    */
    public float getFloatArg(String kernelname,int argn){
        return floatArgs.get(kernelname).get(argn);
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
    @SuppressWarnings("unchecked")
    public ArrayList getBufferAsArrayList(String kernelname,String type ,int argn, int size,boolean print){
        ArrayList arr;
        if(type.equalsIgnoreCase("int")){
            CLBuffer<IntBuffer> buffer = intBuffers.get(kernelname).get(argn);
            // Rewind to keep buffer intact
            arr = new ArrayList<Integer>();
            outputClassMessage("Retrieving "+size+" members of Buffer from kernel: "+
                kernelname+" of "+type+"  in argument:"+argn);
            // push the buffer into array
            int max = 0; int val;
            for(int i = 0; i < size; i++){
                val = buffer.getBuffer().get();
                if( max < val ){max = val;}
                arr.add(i,val);
                if(print){     
                    if(printJustVals){
                        System.out.println(arr.get(i));
                    }else{
                        System.out.println("i: "+i+"   val: "+arr.get(i));
                    }
                }
            }
            lastBufferMaxValue = (double) max;
            // Rewind to keep buffer intact
            buffer.getBuffer().rewind();
        }else if(type.equalsIgnoreCase("float")){
            CLBuffer<FloatBuffer> buffer = flBuffers.get(kernelname).get(argn);
            arr = new ArrayList<Float>();
            outputClassMessage("Retrieving Buffer from kernel: "+
                    kernelname+" of "+type+"  in argument:"+argn);
            // push the buffer into array
            float max = 0; float val;
            for(int i = 0; i < size; i++){
                val = buffer.getBuffer().get();
                if( max < val ){max = val;}
                arr.add(i,val);
                if(print){     
                    if(printJustVals){
                        System.out.println(arr.get(i));
                    }else{
                        System.out.println("i: "+i+"   val: "+arr.get(i));
                    }
                }
            }
            lastBufferMaxValue = (double) max;
            // Rewind to keep buffer intact
            buffer.getBuffer().rewind();
        }else if(type.equalsIgnoreCase("long")){
            CLBuffer<LongBuffer> buffer = longBuffers.get(kernelname).get(argn);
            arr = new ArrayList<Long>();
            outputClassMessage("Retrieving Buffer from kernel: "+
                    kernelname+" of "+type+"  in argument:"+argn);
            // push the buffer into array
            long max = 0; long val;
            for(int i = 0; i < size; i++){
                val = buffer.getBuffer().get();
                if( max < val ){max = val;}
                arr.add(i,val);
                if(print){     
                    if(printJustVals){
                        System.out.println(arr.get(i));
                    }else{
                        System.out.println("i: "+i+"   val: "+arr.get(i));
                    }
                }
            }
            lastBufferMaxValue = (double) max;
            // Rewind to keep buffer intact
            buffer.getBuffer().rewind();
        }else{
            System.err.println("CLArgHelper | Retrieving Buffer from kernel: "+
                kernelname+" of "+type+"  in argument:"+argn);
            System.err.println("CLArgHelper | TRYING TO GET ARRAY LIST OUT OF INCOMPATIBLE BUFFER.");
             arr=null;
        }
        return arr;
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
        double sum = 0.0;
        double val = 0.0;
        if(type.equalsIgnoreCase("int")){
            CLBuffer<IntBuffer> buffer = intBuffers.get(kernelname).get(argn);
            buffer.getBuffer().rewind();
            for(int i = 0; i < size; i++){
                val = buffer.getBuffer().get();
                sum += val;
            }
            // Rewind to keep buffer intact
            buffer.getBuffer().rewind();
        }else if(type.equalsIgnoreCase("float")){
            CLBuffer<FloatBuffer> buffer = flBuffers.get(kernelname).get(argn);
            buffer.getBuffer().rewind();
            for(int i = 0; i < size; i++){
                sum += (double)buffer.getBuffer().get();
            }
            // Rewind to keep buffer intact
            buffer.getBuffer().rewind();
        }else if(type.equalsIgnoreCase("long")){
            CLBuffer<LongBuffer> buffer = longBuffers.get(kernelname).get(argn);
            buffer.getBuffer().rewind();
            for(int i = 0; i < size; i++){
                sum += (double)buffer.getBuffer().get();
            }
            // Rewind to keep buffer intact
            buffer.getBuffer().rewind();
        }else{
            System.err.println("CLArgHelper | Retrieving Buffer from kernel: "+
                kernelname+" of "+type+"  in argument:"+argn);
            System.err.println("CLArgHelper | TRYING TO GET ARRAY LIST OUT OF INCOMPATIBLE BUFFER.");
        }
        return sum;
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
        CLBuffer<IntBuffer> buffer=intBuffers.get(kernelname).get(argn);
        //System.out.println("Buffer object is "+buffer);
        queue.putReadBuffer(buffer, true);
        
        int[] arr = new int[size];

        // push the buffer into array
        int max = 0; int val;
        for(int i = 0; i < size; i++){
            val = buffer.getBuffer().get();
            if( max < val ){max = val;}
            arr[i] = val;
            if(print){     
                if(printJustVals){
                    System.out.println(arr[i]);
                }else{
                    System.out.println("i: "+i+"   val: "+arr[i]);
                }
            }
        }
        lastBufferMaxValue = (double) max;
        
        // Need to rewind to start at same position after reads especially if partial
        buffer.getBuffer().rewind();
        
        return arr;
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
        CLBuffer<FloatBuffer> buffer=flBuffers.get(kernelname).get(argn);
        queue.putReadBuffer(buffer, true);
        float[] arr = new float[size];
        
        // push the buffer into array
        float max = 0; float val;
        for(int i = 0; i < size; i++){
            val = buffer.getBuffer().get();
            if( max < val ){max = val;}
            arr[i] = val;
            if(print){     
                if(printJustVals){
                    System.out.println(arr[i]);
                }else{
                    System.out.println("i: "+i+"   val: "+arr[i]);
                }
            }
        }
        lastBufferMaxValue = (double) max;
        
        // Need to rewind to start at same position after reads especially if partial
        buffer.getBuffer().rewind();
        
        return arr;
    }
    
    /**
    *       getDirectFLBuffer gets the direct float buffer from device after updating.
    * 
    * @param kernelname - kernel to set buffer for  
    * @param argn - argument type entry number
    * @return created buffer
    */
    public FloatBuffer getDirectFlBuffer(String kernelname,int argn){
        CLBuffer<FloatBuffer> buffer=flBuffers.get(kernelname).get(argn);
        queue.putReadBuffer(buffer, true);        
        return buffer.getBuffer();
    }
    
    /**
    *       getDirectIntBuffer gets the direct int buffer from device after updating.
    * 
    * @param kernelname - kernel to set buffer for  
    * @param argn - argument type entry number
    * @return created buffer
    */
    public IntBuffer getDirectIntBuffer(String kernelname,int argn){
       CLBuffer<IntBuffer> buffer=intBuffers.get(kernelname).get(argn);
        queue.putReadBuffer(buffer, true);        
        return buffer.getBuffer();
    }
    
    /**
    *       getDirectLongBuffer gets the direct long buffer from device after updating.
    * 
    * @param kernelname - kernel to set buffer for  
    * @param argn - argument type entry number
    * @return created buffer
    */
    public LongBuffer getDirectLongBuffer(String kernelname,int argn){
        CLBuffer<LongBuffer> buffer=longBuffers.get(kernelname).get(argn);
        queue.putReadBuffer(buffer, true);        
        return buffer.getBuffer();
    }
    
    // utility class to output a string formatted to have class name before
    private void outputClassMessage(String msg){
        if(outputMsg){
        System.out.println("CLArgHelper | "+msg);}
    }
    
    /**
    *       setPrintMode sets the print mode in getBuffer functions
    * 
    * @param md - true if printing just values false if print index and values 
    */
    public void setPrintMode(boolean md){
        printJustVals = md;
    }
}