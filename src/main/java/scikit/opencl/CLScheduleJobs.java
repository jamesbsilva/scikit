package scikit.opencl;

/**
* @(#)  CLScheduleJobs
*/

import java.util.ArrayList;

/**
*       CLScheduleJobs defines jobs for to be used for the CLScheduler class.
* 
* <br>
* 
* @author      James B. Silva <jbsilva @ bu.edu>                 
* @since       2013-07
*/
public class CLScheduleJobs{
    static class kernelJob{
        int dim;
        String kernel;
        int gx = 0;
        int lx = 0;
        int gy = 0;
        int ly = 0;
        int gz = 0;
        int lz = 0;
        public kernelJob(String kin, int g, int l){
            dim = 1;
            kernel = kin;
            gx = g;
            lx = l;
        }       
        public kernelJob(String kin, int g, int l, int g2 , int l2){
            dim = 1;
            kernel = kin;
            gx = g;
            lx = l;
            ly = l2;
            gy = g2;
        }       
        public kernelJob(String kin, int g, int l, int g2 , int l2, int g3, int l3){
            dim = 1;
            kernel = kin;
            gx = g;
            lx = l;
            ly = l2;
            gy = g2;
            gz = g3;
            lz = l3;
        }      
    }
    
    static class flBufferJob{
        String kernel;
        int buffNum;
        String kernelSource;
        int buffNumSource;
        boolean copyBuffer=false;
        int buffSize;
        int glInd = -1;
        float[] buffData;
        ArrayList<Float> buffDataAL;
        String rw;
        boolean fillMode;
        boolean glShared = false;
        public flBufferJob(String skern, int sargn ,String kin, int argn){
            copyBuffer = true;
            kernelSource = skern;
            buffNumSource = sargn;
            buffNum = argn;
            kernel = kin;
        }
        public flBufferJob(String kin,int bn,int bsize,float[] din,String rin, boolean fill){
            kernel = kin;
            buffNum = bn;
            buffSize = bsize;
            buffData = din;
            rw = rin;
            fillMode = fill;
        }
        public flBufferJob(String kin,int bn,int bsize,ArrayList<Float> din,String rin, boolean fill){
            kernel = kin;
            buffNum = bn;
            buffSize = bsize;
            buffDataAL = din;
            rw = rin;
            fillMode = fill;
        }
        public flBufferJob(String kin,int bn,int bsize,float[] din,String rin, boolean fill, int gl){
            kernel = kin;
            buffNum = bn;
            buffSize = bsize;
            buffData = din;
            rw = rin;
            fillMode = fill;
            glInd = gl;
            glShared = true;
        }
        public flBufferJob(String kin,int bn,int bsize,ArrayList<Float> din,String rin, boolean fill,int gl){
            kernel = kin;
            buffNum = bn;
            buffSize = bsize;
            buffDataAL = din;
            rw = rin;
            fillMode = fill;
            glInd = gl;
            glShared = true;
        }
    }
    static class intBufferJob{
        String kernel;
        int buffNum;
        int buffSize;
        String kernelSource;
        int buffNumSource;
        boolean copyBuffer=false;
        int[] buffData;
        ArrayList<Integer> buffDataAL;
        String rw;
        boolean fillMode;
        public intBufferJob(String skern, int sargn ,String kin, int argn){
            copyBuffer = true;
            kernelSource = skern;
            buffNumSource = sargn;
            buffNum = argn;
            kernel = kin;
        }
        public intBufferJob(String kin,int bn,int bsize,int[] din,String rin, boolean fill){
            kernel = kin;
            buffNum = bn;
            buffSize = bsize;
            buffData = din;
            rw = rin;
            fillMode = fill;
        }
        public intBufferJob(String kin,int bn,int bsize,ArrayList<Integer> din,String rin, boolean fill){
            kernel = kin;
            buffNum = bn;
            buffSize = bsize;
            buffDataAL = din;
            rw = rin;
            fillMode = fill;
        }
    }
    
    static class longBufferJob{
        String kernel;
        int buffNum;
        int buffSize;
        String kernelSource;
        int buffNumSource;
        boolean copyBuffer=false;
        long[] buffData;
        ArrayList<Long> buffDataAL;
        String rw;
        boolean fillMode;
        public longBufferJob(String skern, int sargn ,String kin, int argn){
            copyBuffer = true;
            kernelSource = skern;
            buffNumSource = sargn;
            buffNum = argn;
            kernel = kin;
        }
        public longBufferJob(String kin,int bn,int bsize,long[] din,String rin, boolean fill){
            kernel = kin;
            buffNum = bn;
            buffSize = bsize;
            buffData = din;
            rw = rin;
            fillMode = fill;
        }
        public longBufferJob(String kin,int bn,int bsize,ArrayList<Long> din,String rin, boolean fill){
            kernel = kin;
            buffNum = bn;
            buffSize = bsize;
            buffDataAL = din;
            rw = rin;
            fillMode = fill;
        }
    }
}