package scikit.opencl;

/**
* 
*    @(#)   CLScheduler
*/  

import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.media.opengl.GLAutoDrawable;
import scikit.jobs.SimulationCL;
import scikit.opencl.CLScheduleJobs.*;


/**
*       CLScheduler handles timing of events with a GLEventListener in the form created
*   in the BasicEventListenerCL class. This class should be used to schedule buffers made
*   when using OpenCL/OpenGL interoperatibility. In simulation mode an openCL 
*   enabled step can be called by setting a SimulationCL object in set Simulation.
* 
* <br>
* 
* @author      James B. Silva <jbsilva @ bu.edu>                 
* @since       2013-07
*/
public class CLScheduler{
    private CLHelper clhelper;
    private SimulationCL sim; 
    private boolean simMode = false;
    private BasicEventListenerCLGL _bel;
    private int swapInterval = 20;
    private AtomicBoolean scheduledEvent = new AtomicBoolean(false);
    private AtomicBoolean schedBuffer = new AtomicBoolean(false);
    private AtomicBoolean schedKernel = new AtomicBoolean(false);
    private AtomicInteger kernelRunsCompleted = new AtomicInteger(0);
    private String posKernel = ""; private String colKernel = ""; private String initKernel = "";
    private int initKernelPosBuffNum; private int initKernelColBuffNum;
    private int[] kernelPosWorkInfo = new int[5]; private int[] kernelColWorkInfo = new int[5];
    private AtomicBoolean noKernelChunks = new AtomicBoolean(false);
    private AtomicInteger kernelRunChunk = new AtomicInteger(0);
    private ArrayList<flBufferJob> fljobs; private ArrayList<intBufferJob> intjobs;
    private ArrayList<longBufferJob> longjobs;
    private ArrayList<kernelJob> kerneljobs;
    
    /**
    *        CLScheduler constructor
    * 
    *   @param bin - BasicEventListener which the scheduler is coordinating with.
    * 
    */     
    public CLScheduler(BasicEventListenerCLGL bin){
        _bel = bin;
        initKernel = _bel.getSceneKernel();
        initKernelPosBuffNum = _bel.getSceneKernelPosBuffNum();
        initKernelColBuffNum = _bel.getSceneKernelColBuffNum();
        fljobs = new ArrayList<flBufferJob>();
        intjobs = new ArrayList<intBufferJob> ();
        longjobs = new ArrayList<longBufferJob> ();
        kerneljobs = new ArrayList<kernelJob> ();
    }
    
    /**
    *         setCLHelper sets the clhelper to be linked to this CLScheduler class
    *   which can be used for OpenCL calls which do not need to be scheduled because
    *   they are timed with an OpenGL callback function.This calls synchronizes the clhelper
    *   with the clscheduler class clhelper.
    * 
    *  @param clin - clhelper to be used in the  context for this GLEventListener
    */ 
    public void setCLHelper(CLHelper clin){
        clhelper = clin;
    }
    /**
    *         setSimulation sets the SimulationCL that needs to be called and timed with scheduler
    * 
    *  @param si - SimulationCL that needs to be called and timed with scheduler
    */ 
    public void setSimulation(SimulationCL si){
        sim = si;
    }
    /**
    *         setKernelChunkRunsOff sets the scheduler to run continously
    *   or in chunks
    * 
    *  @param us - true if running continuously ( no chunks)
    */ 
    public void setKernelChunkRunsOff(boolean us){
        noKernelChunks.set(us);
    }
    /**
    *         getRunsCompleted returns the amount of kernel run completed.
    * 
    *  @return amount of kernel runs completed (mctime)
    */ 
    public int getRunsCompleted(){
        return kernelRunsCompleted.get();
    }
    /**
    *         getRunsCompleted resets the amount of kernel run completed.
    * 
    */ 
    public void clearRunsCompleted(){
        kernelRunsCompleted.set(0);
    }
    /**
    *         schedCopyPosSharedBuffer schedules copying of the position buffer
    *   into the given float buffer in given float buffer.
    * 
    *  @param kern - kernel to copy the position float buffer into
    *  @param buffnum -float buffer in which kernel to copy the position float buffer into
    */ 
    public void schedCopyPosSharedBuffer(String kern, int buffnum){
        if(schedBuffer.get()){waitForLastScheduled();}
        flBufferJob  fljob = new  flBufferJob(initKernel,initKernelPosBuffNum,kern, buffnum); 
        fljobs.add(fljob);
        schedBuffer.set(true);
        scheduledEvent.set(true);
    }
    /**
    *         schedCopyColSharedBuffer schedules copying of the color buffer
    *   into the given float buffer in given float buffer.
    * 
    *  @param kern - kernel to copy the color float buffer into
    *  @param buffnum -float buffer in which kernel to copy the position float buffer into
    */ 
    public void schedCopyColSharedBuffer(String kern, int buffnum){
        if(schedBuffer.get()){waitForLastScheduled();}
        flBufferJob  fljob = new  flBufferJob(initKernel,initKernelColBuffNum,kern, buffnum); 
        fljobs.add(fljob);
        schedBuffer.set(true);
        scheduledEvent.set(true);
    }
    /**
    *         schedMakePosSharedBuffer schedules making of the position buffer
    *   into the given float buffer in given float buffer.
    * 
    *  @param kern - kernel to copy the position float buffer into
    *  @param buffnum -float buffer in which kernel argument number (must create float buffer 0 before 1 ,1 before 2)
    *  @param buffsize -float buffer size
    *  @param data -float array to initialize buffer into
    *  @param rw - read write ( r/w/rw)
    *  @param fillprop - fill mode for buffer (true if fill sequential copies of data buffer until buffer is filled)
    */ 
    public void schedMakePosSharedBuffer(String kern, int buffnum, int buffsize,
            float[] data,String rw,boolean fillprop){
        if(schedBuffer.get()){waitForLastScheduled();}
        flBufferJob  fljob = new  flBufferJob(kern,buffnum,buffsize,data,rw,fillprop,_bel.getPosGLind()); 
        fljobs.add(fljob);
        schedBuffer.set(true);
        scheduledEvent.set(true);
    }
    /**
    *         schedMakeFloatBuffer schedules making of a float buffer
    *   for the given kernel.
    * 
    *  @param kern - kernel to create the float buffer into
    *  @param buffnum -float buffer in which kernel argument number (must create float buffer 0 before 1 ,1 before 2)
    *  @param buffsize -float buffer size
    *  @param data -float array to initialize buffer into
    *  @param rw - read write ( r/w/rw)
    *  @param fillprop - fill mode for buffer (true if fill sequential copies of data buffer until buffer is filled)
    */ 
    public void schedMakeFloatBuffer(String kern, int buffnum, int buffsize,
            float[] data,String rw,boolean fillprop){
        if(schedBuffer.get()){waitForLastScheduled();}
        flBufferJob  fljob = new  flBufferJob(kern,buffnum,buffsize,data,rw,fillprop); 
        fljobs.add(fljob);
        schedBuffer.set(true);
        scheduledEvent.set(true);
    }

    /**
    *         schedMakeIntBuffer schedules making of a int buffer
    *   for the given kernel.
    * 
    *  @param kern - kernel to create the  buffer into
    *  @param buffnum - buffer in which kernel argument number (must create int buffer 0 before 1 ,1 before 2)
    *  @param buffsize -buffer size
    *  @param data - array to initialize buffer into
    *  @param rw - read write ( r/w/rw)
    *  @param fillprop - fill mode for buffer (true if fill sequential copies of data buffer until buffer is filled)
    */ 
    public void schedMakeIntBuffer(String kern, int buffnum, int buffsize,
            int[] data,String rw,boolean fillprop){
        if(schedBuffer.get()){waitForLastScheduled();}
        intBufferJob  intjob = new  intBufferJob(kern,buffnum,buffsize,data,rw,fillprop); 
        intjobs.add(intjob);
        schedBuffer.set(true);
        scheduledEvent.set(true);
    }
    
    /**
    *         schedMakeLongBuffer schedules making of a long buffer
    *   for the given kernel.
    * 
    *  @param kern - kernel to create the  buffer into
    *  @param buffnum - buffer in which kernel argument number (must create long buffer 0 before 1 ,1 before 2)
    *  @param buffsize -buffer size
    *  @param data - array to initialize buffer into
    *  @param rw - read write ( r/w/rw)
    *  @param fillprop - fill mode for buffer (true if fill sequential copies of data buffer until buffer is filled)
    */ 
    public void schedMakeLongBuffer(String kern, int buffnum, int buffsize,
            long[] data,String rw,boolean fillprop){
        if(schedBuffer.get()){waitForLastScheduled();}
        longBufferJob  longjob = new  longBufferJob(kern,buffnum,buffsize,data,rw,fillprop); 
        longjobs.add(longjob);
        schedBuffer.set(true);
        scheduledEvent.set(true);
    }
    
    /**
    *         schedMakeColSharedBuffer schedules making of the color buffer
    *   into the given float buffer in given float buffer.
    * 
    *  @param kern - kernel to copy the position float buffer into
    *  @param buffnum -float buffer in which kernel argument number (must create float buffer 0 before 1 ,1 before 2)
    *  @param buffsize -float buffer size
    *  @param data -float array to initialize buffer into
    *  @param rw - read write ( r/w/rw)
    *  @param fillprop - fill mode for buffer (true if fill sequential copies of data buffer until buffer is filled)
    */ 
    public void schedMakeColSharedBuffer(String kern, int buffnum, int buffsize,
            float[] data,String rw,boolean fillprop){
        if(schedBuffer.get()){waitForLastScheduled();}
        flBufferJob  fljob = new  flBufferJob(kern,buffnum,buffsize,data,rw,fillprop,_bel.getColGLind()); 
        fljobs.add(fljob);
        schedBuffer.set(true);
        scheduledEvent.set(true);
    }

    /**
    *         schedUpdatePosSharedBuffer schedules update of the position buffer
    *   into the given float buffer in given float buffer.
    * 
    *  @param dataSize -float buffer size
    *  @param data -float array to initialize buffer into
    *  @param fillprop - fill mode for buffer (true if fill sequential copies of data buffer until buffer is filled)
    */ 
    public void schedUpdatePosSharedBuffer(float[] data,int dataSize,boolean fillprop){
        if(schedBuffer.get()){waitForLastScheduled();}
        flBufferJob  fljob = new  flBufferJob(initKernel,initKernelPosBuffNum,dataSize,data,fillprop,_bel.getPosGLind()); 
        fljobs.add(fljob);
        schedBuffer.set(true);
        scheduledEvent.set(true);
    }
    
    /**
    *         schedUpdateIntBuffer schedules updating of the buffer
    *   into the given int buffer in given int buffer.
    * 
    *  @param kern - kernel to copy the position float buffer into
    *  @param buffnum -float buffer in which kernel argument number (must create float buffer 0 before 1 ,1 before 2)
    *  @param dataSize -int buffer size
    *  @param data -int array to initialize buffer into
    *  @param fillprop - fill mode for buffer (true if fill sequential copies of data buffer until buffer is filled)
    */ 
    public void schedUpdateIntBuffer(String kern, int buffnum, int[] data,int dataSize,boolean fillprop){
        if(schedBuffer.get()){waitForLastScheduled();}
        intBufferJob  intjob = new  intBufferJob(kern,buffnum,dataSize,data,fillprop); 
        intjobs.add(intjob);
        schedBuffer.set(true);
        scheduledEvent.set(true);
    }

    /**
    *         schedUpdateIntBuffer schedules updating of the buffer
    *   into the given int buffer in given int buffer.
    * 
    *  @param kern - kernel to copy the position float buffer into
    *  @param buffnum -float buffer in which kernel argument number (must create float buffer 0 before 1 ,1 before 2)
    *  @param dataSize -float buffer size
    *  @param data -float array to initialize buffer into
    *  @param fillprop - fill mode for buffer (true if fill sequential copies of data buffer until buffer is filled)
    */ 
    public void schedUpdateIntBuffer(String kern, int buffnum, ArrayList<Integer> data,int dataSize,boolean fillprop){
        if(schedBuffer.get()){waitForLastScheduled();}
        intBufferJob  intjob = new  intBufferJob(kern,buffnum,dataSize,data,fillprop); 
        intjobs.add(intjob);
        schedBuffer.set(true);
        scheduledEvent.set(true);
    }
    
    /**
    *         schedUpdateFloatBuffer schedules updating of the color buffer
    *   into the given float buffer in given float buffer.
    * 
    *  @param kern - kernel to copy the position float buffer into
    *  @param buffnum -float buffer in which kernel argument number (must create float buffer 0 before 1 ,1 before 2)
    *  @param dataSize -float buffer size
    *  @param data -float array to initialize buffer into
    *  @param fillprop - fill mode for buffer (true if fill sequential copies of data buffer until buffer is filled)
    */ 
    public void schedUpdateFloatBuffer(String kern, int buffnum, ArrayList<Float> data,int dataSize,boolean fillprop){
        if(schedBuffer.get()){waitForLastScheduled();}
        flBufferJob  fljob = new  flBufferJob(kern,buffnum,dataSize,data,fillprop); 
        fljobs.add(fljob);
        schedBuffer.set(true);
        scheduledEvent.set(true);
    }
    /**
    *         schedUpdateFloatBuffer schedules updating of the color buffer
    *   into the given float buffer in given float buffer.
    * 
    *  @param kern - kernel to copy the position float buffer into
    *  @param buffnum -float buffer in which kernel argument number (must create float buffer 0 before 1 ,1 before 2)
    *  @param dataSize -float buffer size
    *  @param data -float array to initialize buffer into
    *  @param fillprop - fill mode for buffer (true if fill sequential copies of data buffer until buffer is filled)
    */ 
    public void schedUpdateFloatBuffer(String kern, int buffnum, float[] data,int dataSize,boolean fillprop){
        if(schedBuffer.get()){waitForLastScheduled();}
        flBufferJob  fljob = new  flBufferJob(kern,buffnum,dataSize,data,fillprop); 
        fljobs.add(fljob);
        schedBuffer.set(true);
        scheduledEvent.set(true);
    }
    
    /**
    *         schedUpdateColSharedBuffer schedules updating of the color buffer
    *   into the given float buffer in given float buffer.
    * 
    *  @param dataSize -float buffer size
    *  @param data -float array to initialize buffer into
    *  @param fillprop - fill mode for buffer (true if fill sequential copies of data buffer until buffer is filled)
    */ 
    public void schedUpdateColSharedBuffer(float[] data,int dataSize,boolean fillprop){
        if(schedBuffer.get()){waitForLastScheduled();}
        flBufferJob  fljob = new  flBufferJob(initKernel,initKernelColBuffNum,dataSize,data,fillprop,_bel.getColGLind()); 
        fljobs.add(fljob);
        schedBuffer.set(true);
        scheduledEvent.set(true);
    }
    
    // runs any scheduled float buffer operations
    private void runSchedFloatBuffer(flBufferJob fljob){
        if(fljob.copyBuffer){
            //System.out.println("Copying from : "+fljob.kernelSource+"    to | "+fljob.kernel);
            clhelper.copyFlBufferAcrossKernel(fljob.kernelSource, fljob.buffNumSource, fljob.kernel, fljob.buffNum);
        }if(fljob.updateBuffer){
            // update buffer based on data
            FloatBuffer tempBuff = (FloatBuffer) clhelper.getArgHandler().getDirectFlBuffer(fljob.kernel, fljob.buffNum);
            if(fljob.buffDataAL != null){
                for(int u = 0; u < fljob.buffDataAL.size();u++){
                    tempBuff.put(u,fljob.buffDataAL.get(u));
                }
            }else{
                for(int u = 0; u < fljob.buffData.length;u++){
                    tempBuff.put(u,fljob.buffData[u]);       
                }
            }
        }else{
            if(fljob.glShared){
                clhelper.createFloatBufferGL(fljob.kernel,fljob.buffNum,
                        fljob.buffSize, fljob.buffData, fljob.rw, fljob.fillMode, fljob.glInd);
            }else{
                clhelper.createFloatBuffer(fljob.kernel,fljob.buffNum,
                            fljob.buffSize, fljob.buffData, fljob.rw, fljob.fillMode);    
            }
        }
    }

    // runs any scheduled int buffer operations
    private void runSchedIntBuffer(intBufferJob intjob){
        if(intjob.updateBuffer){
            // update buffer based on data
            IntBuffer tempBuff = (IntBuffer) clhelper.getArgHandler().getDirectIntBuffer(intjob.kernel, intjob.buffNum);
            if(intjob.buffDataAL != null){
                for(int u = 0; u < intjob.buffDataAL.size();u++){
                    tempBuff.put(u,intjob.buffDataAL.get(u));
                }
            }else{
                for(int u = 0; u < intjob.buffData.length;u++){
                    tempBuff.put(u,intjob.buffData[u]);       
                }
            }
        }else{
            clhelper.createIntBuffer(intjob.kernel,intjob.buffNum,
                    intjob.buffSize, intjob.buffData, clhelper.convertStringToRWInt(intjob.rw), intjob.fillMode);
        }
    }
    
    // runs any scheduled long buffer operations
    private void runSchedLongBuffer(longBufferJob longjob){
        clhelper.createLongBuffer(longjob.kernel,longjob.buffNum,
                    longjob.buffSize, longjob.buffData, clhelper.convertStringToRWInt(longjob.rw), longjob.fillMode);    
    }
    
    // runs any scheduled buffer operations if there are any
    private void makeSchedBuffer(){
        if(schedBuffer.get()){
            schedBuffer.set(false);
            for(int u = 0; u < fljobs.size();u++){
                runSchedFloatBuffer(fljobs.get(u));
            }
            for(int u = 0; u < intjobs.size();u++){
                runSchedIntBuffer(intjobs.get(u));
            }
            for(int u = 0; u < longjobs.size();u++){
                runSchedLongBuffer(longjobs.get(u));
            }
            fljobs.clear();
            intjobs.clear();
            longjobs.clear();
        }
    }
    
    /**
    *         sched1DKernel schedules the running of a 1D kernel
    * 
    *  @param kernel - kernel to schedule the running of 
    *  @param gx - global work size
    *  @param lx -local work size
    */ 
    public void schedule1DKernel(String kernel, int gx , int lx){
        if(schedKernel.get()){waitForLastScheduled();}
        kernelJob kjob = new kernelJob(kernel,gx,lx);
        schedKernel.set(true);
        scheduledEvent.set(true);
    }
    
    /**
    *         sched2DKernel schedules the running of a 2D kernel
    * 
    *  @param kernel - kernel to schedule the running of 
    *  @param gx - global work size
    *  @param lx -local work size
    *  @param gy - global work size 2nd dimension
    *  @param ly -local work size 2nd dimension
    */ 
    public void schedule2DKernel(String kernel, int gx , int lx, int gy , int ly){
        if(schedKernel.get()){waitForLastScheduled();}
        kernelJob kjob = new kernelJob(kernel,gx,lx,gy,ly);
        schedKernel.set(true);
        scheduledEvent.set(true);
    }
    
    /**
    *         schedKernelRunChunk schedules the running of all kernels an amount given by 
    *   the input chunk.
    * 
    *  @param chunk - chunk size
    */ 
    public void scheduleKernelRunChunk(int chunk){
        kernelRunChunk.set(chunk);
        setKernelChunkRunsOff(false);
    }
    // run scheduled running of a kernel given by job
    private void runScheduledKernel(kernelJob kjob){
        if(kjob.dim == 1){
            clhelper.runKernel(kjob.kernel,kjob.gx,kjob.lx);
        }else if(kjob.dim == 2){
            clhelper.runKernel2D(kjob.kernel,kjob.gx,kjob.lx,kjob.gy,kjob.ly);
        }else if(kjob.dim == 3){
            clhelper.runKernel3D(kjob.kernel,kjob.gx,kjob.lx,kjob.gy,kjob.ly,kjob.gz,kjob.lz);
        }
        schedKernel.set(false);
    }
    // run scheduled event if there is any scheduled
    private void runScheduledEvent(){
        if(schedBuffer.get()){
            makeSchedBuffer();
        }
        if(schedKernel.get()){
            schedKernel.set(false);
            for(int u = 0; u < kerneljobs.size();u++){
                runScheduledKernel(kerneljobs.get(u));
            }
        }
        scheduledEvent.set(false);
    }
    // check for any scheduled kernels and run position/color kernels
    private void runKernels() {
        if(kernelRunChunk.get() > 0 || noKernelChunks.get()){
            //System.out.println("Running Kernel EL | runAlways: "+noKernelChunks.get()+"  kernel: "+posKernel);
            kernelRunChunk.decrementAndGet();
            boolean regularRun = false;
            if(sim != null){
                sim.oneCalcStep();
            }
            regularRun = true;
            if(posKernel != "" ){
                runPositionKernel();regularRun = true;      
            }
            if(colKernel != "" && (posKernel != colKernel)){
                //System.out.println("CLMetropolisMC | Running  "+colKernel);
                //clhelper.setKernelArg(colKernel,true);
                runColorKernel();regularRun = true;
            }
            if(regularRun){kernelRunsCompleted.incrementAndGet();}
        }
    }
    
    /**
    *         setColorKernel1D set the kernel that updates color buffer for a 1D kernel
    * 
    *  @param kerin - kernel to set as the kernel that updates color buffer
    *  @param gx - global work size
    *  @param lx -local work size
    */ 
    public void setColorKernel1D(String kerin,int gx, int lx){
        kernelColWorkInfo = new int[5];
        kernelColWorkInfo[0] = 1;
        kernelColWorkInfo[1] = gx;
        kernelColWorkInfo[2] = lx;
        colKernel = kerin;
    }
    
    /**
    *         setColorKernel2D set the kernel that updates color buffer for a 2D kernel
    * 
    *  @param kerin - kernel to set as the kernel that updates color buffer
    *  @param gx - global work size
    *  @param lx -local work size
    *  @param gy - global work size 2nd dimension
    *  @param ly -local work size 2nd dimension
    */ 
    public void setColorKernel2D(String kerin,int gx, int lx, int gy,int ly){
        kernelColWorkInfo = new int[5];
        kernelColWorkInfo[0] = 2;
        kernelColWorkInfo[1] = gx;
        kernelColWorkInfo[2] = lx;
        kernelColWorkInfo[3] = gy;
        kernelColWorkInfo[4] = ly;
        colKernel = kerin;
    }
    
    /**
    *         setPositionKernel1D set the kernel that updates position buffer for a 1D kernel
    * 
    *  @param kerin - kernel to set as the kernel that updates position buffer
    *  @param gx - global work size
    *  @param lx -local work size
    */ 
    public void setPositionKernel1D(String kerin,int gx, int lx ){
        kernelPosWorkInfo = new int[5];
        kernelPosWorkInfo[0] = 1;
        kernelPosWorkInfo[1] = gx;
        kernelPosWorkInfo[2] = lx;
        posKernel = kerin;
    }

    /**
    *         setPositionKernel2D set the kernel that updates position buffer for a 2D kernel
    * 
    *  @param kerin - kernel to set as the kernel that updates position buffer
    *  @param gx - global work size
    *  @param lx -local work size
    */ 
    public void setPosColKernel1D(String kerin,int gx, int lx){
        setPositionKernel1D(kerin, gx, lx);
        setColorKernel1D(kerin, gx, lx);
    }
    
    /**
    *         setPosColKernel1D set the kernel that updates color and position buffer for a 1D kernel
    * 
    *  @param kerin - kernel to set as the kernel that updates color and position buffer
    *  @param gx - global work size
    *  @param lx -local work size
    *  @param gy - global work size 2nd dimension
    *  @param ly -local work size 2nd dimension
    */ 
    public void setPosColKernel2D(String kerin,int gx, int lx,int gy, int ly){
        setPositionKernel2D(kerin, gx, lx,gy,ly);
        setColorKernel2D(kerin, gx, lx,gy,ly);
    }
    
    /**
    *         setPosColKernel2D set the kernel that updates color and position buffer for a 2D kernel
    * 
    *  @param kerin - kernel to set as the kernel that updates color and position buffer
    *  @param gx - global work size
    *  @param lx -local work size
    *  @param gy - global work size 2nd dimension
    *  @param ly -local work size 2nd dimension
    */ 
    public void setPositionKernel2D(String kerin,int gx, int lx , int gy,int ly){
        kernelPosWorkInfo = new int[5];
        kernelPosWorkInfo[0] = 2;
        kernelPosWorkInfo[1] = gx;
        kernelPosWorkInfo[2] = lx;
        kernelPosWorkInfo[3] = gy;
        kernelPosWorkInfo[4] = ly;
        posKernel = kerin;
    }
    // run the color kernel    
    private void runColorKernel() {        
        if(kernelColWorkInfo[0] == 2){
            clhelper.runKernel2D(colKernel, kernelColWorkInfo[1], kernelColWorkInfo[3],
                    kernelColWorkInfo[2], kernelColWorkInfo[4]);
        }else{
            clhelper.runKernel(colKernel, kernelColWorkInfo[1], kernelColWorkInfo[2]);
        }
    }
    // run the position kernel
    private void runPositionKernel() {        
        if(kernelPosWorkInfo[0] == 2){
            clhelper.runKernel2D(posKernel, kernelPosWorkInfo[1], kernelPosWorkInfo[3], 
                    kernelPosWorkInfo[2], kernelPosWorkInfo[4]);
        }else{
            clhelper.runKernel(posKernel, kernelPosWorkInfo[1], kernelPosWorkInfo[2]);
        }
    }   
    /**
    *         checks for any scheduled events and runs kernels. This is an unscheduled call which is 
    *   meant to be made by GLEventListener hence the input parameter
    * 
    *   @param drawable - input drawable from GLEventListener
    */
    public void checkAndRun(GLAutoDrawable drawable){
        if(scheduledEvent.get()){
            runScheduledEvent();  
        }
        runKernels();
    }
    // waits for last scheduled event to finish
    private void waitForLastScheduled(){
        while(scheduledEvent.get()){
            try {
                Thread.sleep(swapInterval);
            } catch (InterruptedException ex) {
                Logger.getLogger(BasicEventListenerCLGL.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }
    
    /**
    *         waitForInitiatedCLGL waits until OpenCL and OpenGL has been properly initialized
    *   as given by the input event listener.
    */
    public void waitForInitiatedCLGL(){
        while(!_bel.getCLGLInitStatus()){
            try {
                Thread.sleep(swapInterval);
            } catch (InterruptedException ex) {
                Logger.getLogger(BasicEventListenerCLGL.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }

}
