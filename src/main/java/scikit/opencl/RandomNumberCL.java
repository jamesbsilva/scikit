package scikit.opencl;

/**
* @(#)  RandomNumberCL
*/

// imports
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

/**
 *      RandomNumberCL manages OpenCL random number generating for a few random 
 *  number generators (Random123, MersenneTwister (NVIDIA implementation),MWC). 
 *  You should install some of the following. Random123 is the default RNG.
 *
 * @see <a href="http://www.deshawresearch.com/resources_random123.html">http://www.deshawresearch.com/resources_random123.html</a> 
 * @see <a href="http://cas.ee.ic.ac.uk/people/dt10/research/rngs-gpu-mwc64x.html">http://cas.ee.ic.ac.uk/people/dt10/research/rngs-gpu-mwc64x.html</a> 
 * @see <a href="https://developer.nvidia.com/opencl">https://developer.nvidia.com/opencl</a>
 * 
 *   <br><br><br>
 *  Usage Example - <br><br>
 *      //  initiate with a random number seed and CLHelper object <br>
 *      RNG = new RandomNumberCL(currentSeed,clhelper); <br>
 *      // want to create nRandom amount of random buffers in float buffer 1 in kernel named kernelName <br>
 *      // returns RNGKey to be used in random number generation <br>
 *      RNGKey = RNG.addRandomBuffer(kernelName, 1, nRandom); <br>
 *      // fill the buffer with random numbers (in this case float buffer 1 for kernel named kernelName) <br>
 *      RNG.fillBufferWithRandom(RNGKey); <br>      
 * 
 * 
 * @author James B. Silva <jbsilva@bu.edu>
 * @since May 12, 2012
 */
public class RandomNumberCL {
    private CLHelper clhandler;
    private String defaultRNG = "random123_interface";
    private String RNGKernel = defaultRNG;
    private String flUpdateKernel = "update_fl_buffer";
    private Random ran = new Random();
    private HashMap<Integer,KernelArgInfo> kernelsInfo;
    private HashMap<KernelArgInfo,Integer> indexInfo;
    private HashMap<Integer,int[]> workgroupInfo; 
    private ArrayList<Float> testSet;
    private int currentBuffer=0;    
    private int nBuffers=0;
    private int maxThreads=129;
    private int nRandom =1;
    private boolean testMode = false;
    private boolean javaRNG = false;
    
    /**
    *         RandomNumberCL constructor. 
    * 
    *  @param seed - seed for random number generator used for OpenCL RNGs
    *  @param cl - clhelper to be used as the context in which to fill with RNGs.
    */ 
    public RandomNumberCL(long seed, CLHelper cl){
        this(seed,cl,1,"");
    }
    /**
    *         RandomNumberCL constructor. 
    * 
    *  @param seed - seed for random number generator used for OpenCL RNGs
    *  @param cl - clhelper to be used as the context in which to fill with RNGs.
    *  @param RNGName - OpenCL RNG to be used
    */ 
    public RandomNumberCL(long seed, CLHelper cl,String RNGName){
        this(seed,cl,1,RNGName);
    }
    /**
    *         RandomNumberCL constructor. 
    * 
    *  @param seed - seed for random number generator used for OpenCL RNGs
    *  @param cl - clhelper to be used as the context in which to fill with RNGs.
    *  @param size - default size of RNG buffer
    *  @param RNGName - OpenCL RNG to be used
    */ 
    public RandomNumberCL(long seed, CLHelper cl, int size, String RNGName){
        nRandom = size;
        kernelsInfo = new HashMap<Integer,KernelArgInfo>();
        indexInfo = new HashMap<KernelArgInfo,Integer>();
        workgroupInfo = new HashMap<Integer,int[]>();
        maxThreads = cl.getCurrentDevice1DMaxWorkItems();
        
        ran.setSeed(seed);
        clhandler = cl;
        
        if(RNGName.equalsIgnoreCase("simple")|| RNGName.equalsIgnoreCase("simple_rng")){
            RNGKernel = "simple_rng";
            System.err.println("Creating RNG | "+RNGKernel+".  Make sure an implementation is available.");
            clhandler.createKernel("", RNGKernel);
        }else if(RNGName.contains("random123") ||RNGName.contains("Random123")){
            RNGKernel = "random123_interface";
            clhandler.createKernelFromSource(RNGKernel,getRandom123InterfaceKernelString());
        }else if (RNGName.contains("mwc64x") ||RNGName.contains("Mwc64x")){
            RNGKernel = "mwc64x_interface";
            clhandler.createKernelFromSource(RNGKernel,getMWCKernelString());
        }else if(RNGName.contains("mersenne") || RNGName.contains("mersenne_twister") 
                || RNGName.contains("mersennetwister")){
            RNGKernel = "mersenne_twister";
            clhandler.createKernelFromSource(RNGKernel,getMersenneKernelString());
        }else{
            RNGName = defaultRNG;
            RNGKernel = defaultRNG;
            clhandler.createKernel("", RNGKernel);
        }
        clhandler.createFloatBuffer(RNGKernel, 0, nRandom, 0);
        
        if(RNGName.equalsIgnoreCase("simple")|| RNGName.equalsIgnoreCase("simple_rng") ){
            setSimpleRNGArg();
        }else if(RNGName.contains("mersenne") || RNGName.contains("mersenne_twister") 
                || RNGName.contains("mersennetwister")){
            setMersenneArg();
        }else if(RNGName.contains("mwc64x") ||RNGName.contains("Mwc64x") ){
            setMwc64xArg();
        }else if(RNGName.contains("random123") ||RNGName.contains("Random123") ){
            setRandom123Arg();
        }
        if(RNGName.equalsIgnoreCase("java")){
            javaRNG=true;
        }
        clhandler.setKernelArg(RNGKernel);
    }
    //set random number cl into a test mode
    private void initializeTestMode(int size){
        clhandler.createKernelFromSource(flUpdateKernel,getFLUpdateKernel());
        clhandler.copyFlBufferAcrossKernel(RNGKernel, 0, flUpdateKernel, 0);
        testSet = new ArrayList<Float>();
        if(size < 0){
            // 8 test values
            /*testSet.add(0.0f);
            testSet.add(0.001f);
            testSet.add(0.25f);
            testSet.add(0.5f);
            testSet.add(0.75f);
            testSet.add(0.99f);
            testSet.add(0.9999f);*/
            testSet.add(0.001f); 
        }else{
            System.out.println("RandomNumberCL | initializing: "+size);
            for(int j=0;j < size;j++){
                testSet.add(ran.nextFloat());
            }
        }
        // started test mode
        testMode=true;
    }
    
    // Fill buffer with numbers using java RNG
    private void fillBufferJavaRNG(int size){
        for(int j=0;j<size;j++){
            testSet.set(j,ran.nextFloat());
        }
    }
    // set seed for random 123
    private void setRandom123Arg(){
        clhandler.setLongArg(RNGKernel, 0, Math.abs(2*ran.nextLong()-(long)(nRandom/4)));
    }
    // set seed for mwc64
    private void setMwc64xArg(){
        clhandler.setLongArg(RNGKernel, 0, Math.abs(2*ran.nextLong()-(long)(nRandom/4)));
    }
    // set seed for mersenne twister
    private void setMersenneArg(){
        clhandler.setIntArg(RNGKernel, 0, Math.abs(2*ran.nextInt()));
        clhandler.setIntArg(RNGKernel, 1, Math.abs(2*ran.nextInt()));
        clhandler.setIntArg(RNGKernel, 2, 1);
    }
    // set seed for simple rng
    private void setSimpleRNGArg(){
        clhandler.setLongArg(RNGKernel, 0, Math.abs(2*ran.nextLong()));
        clhandler.setLongArg(RNGKernel, 1, Math.abs(2*ran.nextLong()));
    }
    // update seed for simple rng
    private void updateSimpleArg(){
        clhandler.setLongArg(RNGKernel, 0, Math.abs(2*ran.nextLong()));
        clhandler.setLongArg(RNGKernel, 1, Math.abs(2*ran.nextLong()));
    }
    // update seed for mersenne twister
    private void updateMersenneArg(){
        clhandler.setIntArg(RNGKernel, 0, Math.abs(2*ran.nextInt()));
        clhandler.setIntArg(RNGKernel, 1, Math.abs(2*ran.nextInt()));
    }
    // update seed for random 123
    private void updateRandom123Arg(){
        clhandler.setLongArg(RNGKernel, 0, Math.abs(2*ran.nextLong()));
    }
    // update seed for mwc64
    private void updateMwc64xArg(){
        clhandler.setLongArg(RNGKernel, 0, Math.abs(2*ran.nextLong()));
    }
    // update seed for random 123 with an offset
    private void updateRandom123Arg(int off){
        clhandler.setLongArg(RNGKernel, 0, Math.abs(2*ran.nextLong()-(long)(off)));
    }
    // update seed for mwc64 with offset
    private void updateMwc64xArg(int off){
        clhandler.setLongArg(RNGKernel, 0, Math.abs(2*ran.nextLong()-(long)(off)));
    }
    /**
    *         setMaxThreads sets the maximum amount of threads/work items
    *   to be run on OpenCL device when generating random numbers.
    * 
    *  @param u - max threads to use when running random buffer kernel
    */ 
    public void setMaxThreads(int u){
        maxThreads=u;
    }
    /**
    *         setSeed sets the seed of the random generator which seeds
    *   the OpenCL random number generators.
    * 
    *  @param seedIn - input seed long
    */ 
    public void setSeed(long seedIn){
        ran.setSeed(seedIn);
    }
    /**
    *         getIndexRandomBuffer searches for the integer index which is used in the fillBuffer 
    *   method. 
    * 
    *  @param kernel - kernel to search for
    *  @param arg - buffer number of search kernel
    *  @return index of the queried buffer to be used with fill buffer method
    */ 
    public int getIndexRandomBuffer(String kernel, int arg){
        KernelArgInfo temp = new KernelArgInfo(kernel,arg);
        return indexInfo.get(temp);
    }
    /**
    *         fillWithTestRandom fills the kernel with a set of random numbers then update the buffer.
    * 
    *  @param buffToFill - kernel in which to fill a buffer with random float [0,1]
    */ 
    public void fillWithTestRandom(int buffToFill){
        if(testMode!= true){
            initializeTestMode(-20);
        }
        
        if(currentBuffer !=buffToFill){
            KernelArgInfo temp = kernelsInfo.get(buffToFill);
            String inKernel = temp.getKernelName();
            int inArgN = temp.getArg();
            clhandler.copyFlBufferAcrossKernel(inKernel, inArgN, RNGKernel, 0,true);
            clhandler.setKernelArg(RNGKernel,true);
            currentBuffer = buffToFill;
        }
        
        int[] workInfo = workgroupInfo.get(buffToFill);
        // fill with test set and update
        clhandler.setFloatBuffer(flUpdateKernel, 0, workInfo[0], testSet, true);
        clhandler.setKernelArg(flUpdateKernel, true); 
        clhandler.runKernel(flUpdateKernel,workInfo[0],workInfo[1]);
    }
    
    // Class divides the work for the running of the kernel
    private void divideWork(int size){
        int[] temp = new int[2];
        temp[0] = size;
        temp[1] = clhandler.maxLocalSize1D(size);
        workgroupInfo.put(nBuffers, temp);
    }
    
    /**
    *         addRandomBuffer 
    * 
    *  @param inKernel - kernel in which to fill a buffer with random float [0,1]
    *  @param inArgN - buffer number of the float buffer 
    *                   that is going to be filled in the inkernel.
    *  @param buffsize - size of the buffer to fill
    *  @return number of buffers currently being filled with random numbers by this class 
    */ 
    public int addRandomBuffer(String inKernel, int inArgN, int buffsize){
        KernelArgInfo temp = new KernelArgInfo(inKernel,inArgN);
        kernelsInfo.put(nBuffers, temp);
        indexInfo.put(temp, nBuffers);
        if(RNGKernel.contains("random123") || RNGKernel.contains("mwc64x")){
            int div = buffsize%4;
            div = (int)(buffsize/4.0)+div;
            divideWork(div);
        }else{
            divideWork(buffsize);
        }
        
        clhandler.copyFlBufferAcrossKernel(inKernel, inArgN, RNGKernel, 0,true);
        clhandler.setKernelArg(RNGKernel,true);
        currentBuffer = nBuffers;
        
        if(javaRNG){
            initializeTestMode(buffsize);
            clhandler.copyFlBufferAcrossKernel(inKernel, inArgN, flUpdateKernel, 0,true);
        }
        nBuffers++;
        return (nBuffers-1);
    }
    
    /**
    *         fillWithTestRandom fills the kernel with a set of random numbers.
    * 
    *  @param buffToFill - index number for kernel/buffer in which to fill a buffer with random float [0,1]
    */ 
    public void fillBufferWithRandom(int buffToFill){
        if(currentBuffer !=buffToFill){
            KernelArgInfo temp = kernelsInfo.get(buffToFill);
            String inKernel = temp.getKernelName();
            int inArgN = temp.getArg();
            clhandler.copyFlBufferAcrossKernel(inKernel, inArgN, RNGKernel, 0,true);
            clhandler.setKernelArg(RNGKernel,true);
            currentBuffer = buffToFill;
        }
        
        int[] workInfo = workgroupInfo.get(buffToFill);
        
        if(javaRNG){
            fillBufferJavaRNG(workInfo[0]);
        
            // fill with test set and update
            clhandler.setFloatBuffer(flUpdateKernel, 0, workInfo[0], testSet, true);
            clhandler.setKernelArg(flUpdateKernel, true); 
            clhandler.runKernel(flUpdateKernel,workInfo[0],workInfo[1]);
        }else{
            // set seeds
            if(RNGKernel.equalsIgnoreCase("simple_rng")){
                updateSimpleArg();
            }else if(RNGKernel.contains("random123")){
                updateRandom123Arg(workInfo[0]);
            }else if(RNGKernel.contains("mersenne")){
                updateMersenneArg();
            }else if(RNGKernel.contains("mwc64x")){
                updateMwc64xArg(workInfo[0]);
            }    
            clhandler.setKernelArg(RNGKernel, true); 
            clhandler.runKernel(RNGKernel,workInfo[0],workInfo[1]);
        }
    }
    /**
    *         printRandom prints generated random numbers.
    * 
    *  @param n - number of random numbers to generate
    */ 
    public void printRandom(int n){
        if(n>nRandom){
            System.err.println("RandomNumberCL | ATTEMPTING TO PRINT MORE "
                    + "RANDOM NUMBERS THAN AVAILABLE. DEFAULTING TO MAX");
            n=nRandom;
        }    
        if(javaRNG){
            fillBufferJavaRNG(n);
            // fill with test set and update
            clhandler.setFloatBuffer(flUpdateKernel, 0, n, testSet, true);
            clhandler.setKernelArg(flUpdateKernel, true); 
            clhandler.runKernel(flUpdateKernel,n,1);
        }else if(RNGKernel.contains("random123") || RNGKernel.contains("mwc64x")){
            int div = n%4;
            div = (int)(n/4.0)+div;
            clhandler.setLongArg(RNGKernel, 0, Math.abs(2*ran.nextInt()-div));
            System.out.println("IN1");
            clhandler.setKernelArg(RNGKernel, true);
            System.out.println("IN2");
            clhandler.runKernel(RNGKernel,div,1);
        }else{
            // set seeds
            clhandler.setIntArg(RNGKernel, 0, Math.abs(2*ran.nextInt()));
            clhandler.setIntArg(RNGKernel, 1, Math.abs(2*ran.nextInt()));
            clhandler.setKernelArg(RNGKernel, true); 
            clhandler.runKernel(RNGKernel,n,1);
        }
        clhandler.getFloatBufferAsArray(RNGKernel, 0, n, true);
        //System.out.println("seedIn: "+clhandler.getLongArg(RNGKernel, 0));
        if(RNGKernel.contains("rando435m123")){
            System.out.println("RandomNumberCL | Integer Random : ");
            clhandler.getIntBufferAsArray(RNGKernel, 0, n, true);
        }
    }
    // Packaging class for kernel information
    private class KernelArgInfo{
        private String KernelName;
        private int Argument;
        public KernelArgInfo(String k,int u){
            KernelName= k;
            Argument = u;
        }
        public void setInfo(String k,int u){
            KernelName= k;
            Argument = u;
        }
        public int getArg(){
            return Argument;
        }
        public String getKernelName(){
            return KernelName;
        }
    }
    // test the class
    public static void main(String[] args) {
        Random r = new Random();
        String rng = "random123";
        int numberToTest = 100000;
        int timesGen = 1;
        CLHelper clhandler = new CLHelper();
        String home = System.getProperty("user.home");
        clhandler.addKernelSearchDir(home+"/NetBeansProjects/GPUKernels");
        clhandler.initializeOpenCL("GPU");  
        RandomNumberCL rand = new  RandomNumberCL(r.nextLong(),clhandler,numberToTest,rng);
        if(rng.contains("java")){rand.initializeTestMode(numberToTest);}
        long t0 = System.nanoTime();
        for(int u = 0; u < timesGen;u++){
            rand.printRandom(numberToTest);
        }
        t0 = System.nanoTime()-t0;
        // 25k generation time - random123 ~ 1180
        // 25k generation time - mersenne  ~ 4340
        // 25k generation time - simple ~ 1150
        // 25k generation time - java ~ 14700 
        System.out.println("RandomNumberCL | Generated "+numberToTest+"  random numbers with "+rng+" in "+(((double)t0)/1000000)+" ms");
        clhandler.closeOpenCL();
    }
    
    public String getRandom123InterfaceKernelString(){
        return 
            "/*\n" +
            "Copyright 2010-2011, D. E. Shaw Research.\n" +
            "All rights reserved.\n" +
            "\n" +
            "Redistribution and use in source and binary forms, with or without\n" +
            "modification, are permitted provided that the following conditions are\n" +
            "met:\n" +
            "\n" +
            "* Redistributions of source code must retain the above copyright\n" +
            "  notice, this list of conditions, and the following disclaimer.\n" +
            "\n" +
            "* Redistributions in binary form must reproduce the above copyright\n" +
            "  notice, this list of conditions, and the following disclaimer in the\n" +
            "  documentation and/or other materials provided with the distribution.\n" +
            "\n" +
            "* Neither the name of D. E. Shaw Research nor the names of its\n" +
            "  contributors may be used to endorse or promote products derived from\n" +
            "  this software without specific prior written permission.\n" +
            "\n" +
            "THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS\n" +
            "\"AS IS\" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT\n" +
            "LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR\n" +
            "A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT\n" +
            "OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,\n" +
            "SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT\n" +
            "LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,\n" +
            "DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY\n" +
            "THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT\n" +
            "(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE\n" +
            "OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.\n" +
            "*/\n" +
            "/*\n" +
            " * This file is the OpenCL kernel.  It gets preprocessed, munged\n" +
            " * into a C string declaration and included in pi_opencl, so that\n" +
            " * running the compiled pi_opencl does not depend on any include\n" +
            " * files, paths etc.\n" +
            " */\n" +
            "\n" +
            "#include <Random123/threefry.h>\n" +
            "\n" +
            "__kernel void random123_interface(__global float* randFl, unsigned long seed ) {\n" +
            "    unsigned tid = get_global_id(0);\n" +
            "    seed = seed+tid;\n" +
            "    threefry4x32_key_t k = {{tid, 0xdecafbad, 0xfacebead, 0x12345678}};\n" +
            "    threefry4x32_ctr_t c = {{seed, 0xf00dcafe, 0xdeadbeef, 0xbeeff00d}};\n" +
            "    union {\n" +
            "        threefry4x32_ctr_t c;\n" +
            "        int4 i;\n" +
            "    } u;\n" +
            "    c.v[0]++;\n" +
            "    u.c = threefry4x32(c, k);\n" +
            "    long x1 = u.i.x, x3 = u.i.y;\n" +
            "    long x2 = u.i.z, x4 = u.i.w;\n" +
            "\n" +
            "    randFl[4*tid] = ((float)x1 + 2147483648.0f) / 4294967295.0f;\n" +
            "    randFl[4*tid+1] = ((float)x2 + 2147483648.0f) / 4294967295.0f;\n" +
            "    randFl[4*tid+2] = ((float)x3 + 2147483648.0f) / 4294967295.0f;\n" +
            "    randFl[4*tid+3] = ((float)x4 + 2147483648.0f) / 4294967295.0f;\n" +
            "}";
    }
    
    public String getMersenneKernelString(){
        return "/*\n" +
            " * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.\n" +
            " *\n" +
            " * Please refer to the NVIDIA end user license agreement [EULA] associated\n" +
            " * with this source code for terms and conditions that govern your use of\n" +
            " * this software. Any use, reproduction, disclosure, or distribution of\n" +
            " * this software and related documentation outside the terms of the EULA\n" +
            " * is strictly prohibited.\n" +
            " *\n" +
            " */\n" +
            "\n" +
            "#define   MT_RNG_COUNT 4096\n" +
            "#define   MT_MM 9\n" +
            "#define   MT_NN 19\n" +
            "#define   MT_WMASK 0xFFFFFFFFU\n" +
            "#define   MT_UMASK 0xFFFFFFFEU\n" +
            "#define   MT_UMASK 0xFFFFFFFEU\n" +
            "\n" +
            "#define      MT_LMASK 0x1U\n" +
            "#define      MATRIX_A 0x9908b0df /* Constant vector a */\n" +
            "#define      MT_SHIFT0 12\n" +
            "#define      MT_SHIFTB 7\n" +
            "#define	     MASK_B 0x9d2c5680\n" +
            "#define      MT_SHIFTC 15\n" +
            "#define      MASK_C 0xefc60000\n" +
            "#define      MT_SHIFT1 18\n" +
            "#define PI 3.14159265358979f\n" +
            "\n" +
            "////////////////////////////////////////////////////////////////////////////////\n" +
            "// OpenCL Kernel for Mersenne Twister RNG [Modified from NVIDIA Example]\n" +
            "////////////////////////////////////////////////////////////////////////////////\n" +
            "__kernel void mersenne_twister(__global float* d_Rand,\n" +
            "			      unsigned int  seed, unsigned int seed2 ,\n" +
            "			      int nPerRng)\n" +
            "{\n" +
            "    int globalID = get_global_id(0);\n" +
            "\n" +
            "    int iState, iState1, iStateM, iOut;\n" +
            "    unsigned int mti, mti1, mtiM, x;\n" +
            "    unsigned int mt[MT_NN]; \n" +
            "\n" +
            "    //Initialize current state\n" +
            "    mt[0] = seed+((globalID+seed2)%globalID);\n" +
            "    for (iState = 1; iState < MT_NN; iState++)\n" +
            "        mt[iState] = (1812433253U * (mt[iState - 1] ^ (mt[iState - 1] >> 30)) + iState) & MT_WMASK;\n" +
            "\n" +
            "    iState = 0;\n" +
            "    mti1 = mt[0];\n" +
            "    for (iOut = 0; iOut < (nPerRng+3); iOut++) {\n" +
            "        iState1 = iState + 1;\n" +
            "        iStateM = iState + MT_MM;\n" +
            "        if(iState1 >= MT_NN) iState1 -= MT_NN;\n" +
            "        if(iStateM >= MT_NN) iStateM -= MT_NN;\n" +
            "        mti  = mti1;\n" +
            "        mti1 = mt[iState1];\n" +
            "        mtiM = mt[iStateM];\n" +
            "\n" +
            "        // MT recurrence\n" +
            "        x = (mti & MT_UMASK) | (mti1 & MT_LMASK);\n" +
            "	    x = mtiM ^ (x >> 1) ^ ((x & 1) ? MATRIX_A : 0);\n" +
            "\n" +
            "        mt[iState] = x;\n" +
            "        iState = iState1;\n" +
            "\n" +
            "        //Tempering transformation\n" +
            "        x ^= (x >> MT_SHIFT0);\n" +
            "        x ^= (x << MT_SHIFTB) & MASK_B;\n" +
            "        x ^= (x << MT_SHIFTC) & MASK_C;\n" +
            "        x ^= (x >> MT_SHIFT1);\n" +
            "\n" +
            "        //Convert to (0, 1] float and write to global memory\n" +
            "        //d_Rand[globalID] = ((float)x + 1.0f) / 4294967296.0f;  \n" +
            "        d_Rand[globalID] = ((float)x) / 4294967295.0f;    \n" +
            "    }\n" +
            "\n" +
            "}\n";
    }
    public String getMWCKernelString(){
        return "#include \"mwc64x.cl\"\n" +
            "__kernel void mwc64x_interface(__global float *rand,unsigned long baseOffset){\n" +
            "    int tid = get_global_id(0);\n" +
            "    baseOffset = baseOffset+tid;\n" +
            "    mwc64xvec4_state_t rng;\n" +
            "    MWC64XVEC4_SeedStreams(&rng, baseOffset, 2);\n" +
            "    ulong4 x=convert_ulong4(MWC64XVEC4_NextUint4(&rng));\n" +
            "    rand[4*tid]   = ((float)x.x)/4294967295.0;\n" +
            "    rand[4*tid+1] = ((float)x.y)/4294967295.0;\n" +
            "    rand[4*tid+2] = ((float)x.z)/4294967295.0;\n" +
            "    rand[4*tid+3] = ((float)x.w)/4294967295.0;\n" +
            "}\n" +
            "";
    }
    
    public String getFLUpdateKernel(){
        return "__kernel void update_fl_buffer(__global float *A) {\n" +
            "    return;\n" +
            "}\n" +
            "";
    }
}

