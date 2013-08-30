package scikit.graphics.dim3;

/**
* @(#)  Grid3DCL
*/

import static scikit.util.Utilities.format;

import java.awt.Color;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;
import javax.media.opengl.GL2;
import javax.media.opengl.GLAutoDrawable;
import javax.media.opengl.GLCapabilities;
import javax.media.opengl.GLContext;
import javax.media.opengl.GLProfile;
import javax.media.opengl.awt.GLJPanel;
import javax.media.opengl.glu.gl2.GLUgl2;
import javax.swing.JCheckBoxMenuItem;
import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JMenuItem;

import scikit.graphics.ColorChooser;
import scikit.graphics.ColorGradient;
import scikit.graphics.Drawable;
import scikit.graphics.dim2.Gfx2D;
import scikit.graphics.dim2.Gfx2DSwing;
import scikit.opencl.BasicEventListenerCLGL;
import scikit.opencl.CLHelper;
import scikit.opencl.Scene2DCL;
import scikit.opencl.BasicUSI;
import scikit.util.Bounds;
import scikit.util.DoubleArray;
import scikit.util.FileUtil;

/**
*       Grid3DCL is a lattice creating class with OpenGL/OpenCL interoperability
*   useful as a grid using OpenCL to update the state of the grid.
* 
* <br>
* 
* @author      James B. Silva <jbsilva @ bu.edu>                 
* @since       2013-07
*/
public class Grid3DCL extends Scene2DCL {
        private final GLUgl2 glu = new GLUgl2();
	private final BasicUSI usi;
        private ColorChooser _colors = new ColorGradient();
	private BufferedImage _image = null;
	private int _w = 600, _h = 600;
	private int maxFrameLength = 800;
        private int Lx;private int Ly;private int Lz;
	private double scaleLat = 0.09;
        private double[] _data = null;
        private int[] _pixelArray = null;
        private boolean _autoScale = true;
        private boolean _drawRange = false;
        private boolean clReady = false;
        private double _lo = 0, _hi = 1;
        private String drawLatticeKernel = "create3DLatticeCubic";
        private int posGlInd = 0;
        private int colGlInd = 0;
        private int posBuffNum = 0;
        private int colBuffNum = 1;
        private String defaultDeviceType = "GPU";
        private GLJPanel _canvas;        
        private GLContext _context;        
        private JFrame _frame;
    
        /**
        *         gridCL constructor.
        * 
        *  @param title - title of grid
        *  @param clin - clhelper in which to setup the grid
        *  @param Lxin - x coordinate size of grid
        *  @param Lyin - y coordinate size of grid
        *  @param Lzin - z coordinate size of grid
        *  @param drawScale - adjust scale for drawing lattice
        */ 
        public Grid3DCL(String title,CLHelper clin, int Lxin, int Lyin,int Lzin, double drawScale) {
                super(title,clin);
                initDrawKernel = drawLatticeKernel;
                initDrawKernelPosBufferNum = posBuffNum;
                initDrawKernelColBufferNum = colBuffNum;
                Lx = Lxin;
                Ly = Lyin;
                Lz = Lzin;
                scaleLat = drawScale;
                makeEventListener();
                usi = new BasicUSI();
                if(_component == null)makeFrame(_canvas);            
        }


        /**
        *         gridCL constructor.
        * 
        *  @param title - title of grid
        *  @param clin - clhelper in which to setup the grid
        *  @param Lxin - x coordinate size of grid
        *  @param Lyin - y coordinate size of grid
        *  @param Lzin - z coordinate size of grid
        */ 
        public Grid3DCL(String title,CLHelper clin, int Lxin, int Lyin,int Lzin) {
            // default draw scale of 0.09
            this(title,clin,Lxin,Lyin,Lzin,0.09);
        }
    
        // make the GLEventListener for this grid
        private void makeEventListener(){
            System.out.println("Grid Dimensions | Lx: "+Lx+"  Ly: "+Ly+"  Lz: "+Lz+"   width: "+_w+"   height: "+_h);
            _bel = new BasicEventListenerCLGL(null, 4*Lx*Ly*Lz,4*Lx*Ly*Lz,this,_w,_h);
            clscheduler = _bel.getCLScheduler();
            _canvas.addGLEventListener(_bel);
        }

        /**
        *         isOpenCLReady returns true if CLHelper is properly
        *   setup for interoperability. Any calls with CLHelper on a CLGL context
        *   before this will not cause good behavior.
        * 
        *  @return true - if CLGL is properly initiated
        */ 
        public boolean isOpenCLReady(){
            return _bel.getCLGLInitStatus();
        }
        
	public void clear() {
            // remove data first because super.clear() will cause a drawAll() operation
            _w = _h = 0;
            _image = null;
            _data = null;
            super.clear();
	}

	public void setColors(ColorChooser colors) {
            _colors = colors;
	}
	
	public void setAutoScale() {
            _autoScale = true;
	}
	
	public void setScale(double lo, double hi) {
            _autoScale = false;
            _lo = lo;
            _hi = hi;
	}
	
	public void setDrawRange(boolean b) {
            _drawRange = b;
	}
	
        private void assertGridDim(int w, int h){
            if(w != Lx || h != Ly){
                throw new IllegalArgumentException("Data Array length " + (w*h)
                                + " does not fit specified grid shape (" + Lx + "*" + Ly + ")");

            }
        }
        
        /**
        *         getEventListener returns the GLEventListener derived from 
        *   BasicEventListenerCL which will be used in this grid.
        * 
        *  @return GLEventListener being used for grid
        */ 
        public BasicEventListenerCLGL getEventListener(){
            return _bel;
        }
        
        /**
        *         getCLHelperGrid returns the CLHelper used by this grid
        * 
        *  @return CLHelper used by this grid
        */ 
        public CLHelper getCLHelperGrid(){
            return clhelper;
        }
                
	// Override getImage() to return the "native" pixel-map image
	public BufferedImage getImage(int width, int height) {
		return _image;
	}
	
	public BufferedImage getImage() {
		return _image;
	}
       
	@Override
        protected JComponent createCanvas() {
            //return GLJPanel canvas
            GLJPanel canvas = basicInitGL();
            return canvas;
	}
        
	private void makeFrame(GLJPanel canvas){
            System.out.println("GridCL |  Making Frame");
            JFrame frame = new JFrame("GridCL | "+getTitle());
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.add(canvas);
            frame.setSize(_w, _h);
            frame.setVisible(true);
            _frame = frame;
        }
        
        // basic OpenGL init 
        private GLJPanel basicInitGL(){
            GLCapabilities config = new GLCapabilities(GLProfile.get(GLProfile.GL2));
            config.setSampleBuffers(true);
            config.setNumSamples(4);            
            System.out.println("GridCL | Init GL Canvas");
            GLJPanel canvas = new GLJPanel(config);
            _canvas = canvas;
            return canvas;
        }
        
        /**
        *         initCLGL initiates the JOCL/JOGL in the CLHelper to be used.
        * 
        */ 
        @Override
        public void initCLGL(GLAutoDrawable drawable){
            _context = drawable.getContext();
            System.out.println("GridCL | Initializing OpenCL");
            //System.out.println("GL Info:   Context: "+drawable.getContext());
            clhelper.initializeOpenCL(defaultDeviceType, true, drawable.getContext());
            posGlInd = _bel.getPosGLind();
            colGlInd = _bel.getColGLind();
            //clearMouseListeners();
            usi.init(_canvas);
            _bel.setCLhelper(clhelper);
            initLat();
            System.out.println("GridCL | Done Init Pre Buff");
        }
        
        private void clearMouseListeners(){
            MouseListener[] mls = _canvas.getMouseListeners();
            for(int u = 0; u < mls.length;u++){
                _canvas.removeMouseListener(mls[u]);
            }
        }
        
        // initial drawing of lattice
        private void initLat(){
            int posBuffSize = 4*Lx*Ly*Lz;
            int colBuffSize = 4*Lx*Ly*Lz;
            float[] din = new float[posBuffSize];
            float[] din2 = new float[colBuffSize];
            clhelper.createKernelFromSource(initDrawKernel,getDrawKernelAsString());
            clhelper.createFloatBufferGL(initDrawKernel, initDrawKernelPosBufferNum, posBuffSize, din, "rw", true, _bel.getPosGLind());
            clhelper.createFloatBufferGL(initDrawKernel, initDrawKernelColBufferNum,  colBuffSize, din2, "rw", true, _bel.getColGLind());    
            clhelper.setIntArg(initDrawKernel, 0, Lx);
            clhelper.setIntArg(initDrawKernel, 1, Ly);
            clhelper.setIntArg(initDrawKernel, 2, Lz);
            clhelper.setFloatArg(initDrawKernel, 0, (float)scaleLat);
            clhelper.setKernelArg(initDrawKernel);
            clhelper.runKernel(initDrawKernel, (Lx*Ly*Lz), clhelper.maxLocalSize1D(Lx*Ly*Lz));    
        }
        
        /**
        *         rescaleGridDrawing rescales the distance between the grid.
        * 
        *  @param scale - new drawing scale
        */ 
        public void rescaleGridDrawing(double scale) {
            scaleLat = scale;
            clhelper.setFloatArg(initDrawKernel, 0, (float)scaleLat);
            clhelper.runKernel(initDrawKernel, (Lx*Ly*Lz), clhelper.maxLocalSize1D(Lx*Ly*Lz));
        }
        
        
	@Override
        protected void drawBackground(Gfx2D g) {
            // looks better without background
	}
	
	@Override
        protected List<Drawable<Gfx2D>> getAllDrawables() {
		List<Drawable<Gfx2D>> ds = new ArrayList<Drawable<Gfx2D>>();
		ds.add(_gridDrawable);
		ds.addAll(super.getAllDrawables());
		return ds;
	}
	
	@Override
        protected List<JMenuItem> getAllPopupMenuItems() {
		List<JMenuItem> ret = new ArrayList<JMenuItem>(super.getAllPopupMenuItems());

		JMenuItem menuItem = new JCheckBoxMenuItem("Display range", _drawRange);
		menuItem.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				_drawRange = !_drawRange;
				animate();
			}
		});
		ret.add(menuItem);	
		if (_data != null) {
			menuItem = new JMenuItem("Save grid data ...");
			menuItem.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent e) {
					saveData("grid.txt");
				}
			});
			ret.add(menuItem);
			menuItem = new JMenuItem("Save image ...");
			menuItem.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent e) {
					saveImage("grid.png");
				}
			});
			ret.add(menuItem);
		}
		return ret;
	}
	
	private void findRange() {
            if (_autoScale) {
                _lo = DoubleArray.min(_data);
                _hi = DoubleArray.max(_data);
            }
	}
	
	private Drawable<Gfx2D> _gridDrawable = new Drawable<Gfx2D>() {
            public void draw(Gfx2D g) {
            if (_image != null) {
                ((Gfx2DSwing)g).drawImage(_image, 0, 0, 1, 1);
                if (_drawRange) {
                    g.setProjection(g.pixelBounds()); // draw strings at fixed pixels
                    String str1 = "lo = "+format(_lo);
                    String str2 = "hi = "+format(_hi);
                    double border = 4;
                    double offset = 4;
                    double w = Math.max(g.stringWidth(str1), g.stringWidth(str2));
                    double h = g.stringHeight("");
                    g.setColor(new Color(1f, 1f, 1f, 0.5f));
                    g.fillRect(offset, offset, 2*border+w, 3*border+2*h);
                    g.setColor(Color.BLACK);
                    g.drawString(str1, offset+border, offset+h+2*border);
                    g.drawString(str2, offset+border, offset+border);
                    g.setProjection(g.viewBounds()); // return to scene coordinates
                }
            }
            }
            public Bounds getBounds() {
                return new Bounds(0, 1, 0, 1);
            }
	};
      
	private void saveData(String fname) {
            try {
                fname = FileUtil.saveDialog(_component, fname);
                if (fname != null) {
                    PrintWriter pw = FileUtil.pwFromString(fname);
                    FileUtil.writeOctaveGrid(pw, _data, _w, 1);
                    pw.close();
                }
            } catch (IOException e) {}
	}

	private void saveImage(String fname) {
            try {
                fname = FileUtil.saveDialog(_component, fname);
                if (fname != null) {
                        ImageIO.write(_image, "png", new File(fname));
                }
            } catch (IOException e) {}
	}
        /**
        *         interact is an interface method with UI like the BasicUSI and OpenGL. 
        * 
        *   @param gl - GL2 obj
        */
        public void interact(GL2 gl){
           usi.interact(gl);
        }
        
        @Override
        protected void drawAll(Object g) {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }
        /**
        *         setPointSize sets the size of points drawn.
        * 
        *  @param pointS - size of points to be drawn
        */ 
        public void setPointSize(float pointS) {
            _bel.setPointSize(pointS);
        }
        
        private String getDrawKernelAsString(){
            return
                    "float4 getPosCubic(int i, int j, int k, float scale, int Lx, int Ly, int Lz);\n" +
                    "__kernel void create3DLatticeCubic(__global float4 * latPos,__global float4 *latCol, int Lx, int Ly, int Lz, float scale){\n" +
                    "    unsigned int tId = get_global_id(0);\n" +
                    "    int i = (tId%Lx);\n" +
                    "    int j = ((int)((float)tId/(float)Lx))%Ly;\n" +
                    "    int k = ((int)((float)tId/(float)(Lx*Ly)))%Lz;\n" +
                    "    // calculate uv coordinates\n" +
                    "    latPos[tId] = getPosCubic(i,j,k,scale,Lx,Ly,Lz);\n" +
                    "    return;\n" +
                    "}\n" +
                    "float4 getPosCubic(int i, int j, int k, float scale, int Lx, int Ly, int Lz){\n" +
                    "return (float4)((i-(Lx/2.0))/(Lx*scale), (k-(Lz/2.0))/(Lz*scale), (j-(Ly/2.0))/(Ly*scale), 1.0f);\n" +
                    "}\n";
        }
        
}
