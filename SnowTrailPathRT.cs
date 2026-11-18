using UnityEngine;
using UnityEngine.Rendering;

// 雪地痕迹（路径距离场版）
// 不再往 RT 上盖章，而是每帧从整条路径重算一次带符号距离场：
//   中心为负（凹陷）、外侧为正（雪堆），再交给 compute 生成痕迹与隆起。
[ExecuteAlways]
public class SnowTrailPathRT : MonoBehaviour
{
    [Header("① 雪地与区域")]
    // 雪地材质所在物体，痕迹纹理通过它的材质属性喂进去
    public Renderer snowTarget;
    // 勾上则用手填的区域，否则自动跟随玩家
    public bool useFixedArea = false;
    // 距离场覆盖的世界边长（米）
    public float fieldSize = 10f;
    public Vector3 fieldCenter = Vector3.zero;
    public bool followPlayer = true;

    [Header("② RenderTexture")]
    [Range(128, 2048)] public int resolution = 512;

    [Header("③ 路径")]
    [Range(8, 256)] public int maxPoints = 128;

    // 移动超过这个距离才记一个路径点
    public float pointSpacing = 0.08f;

    [Header("④d 恢复（痕迹被雪慢慢填平）")]
    [Range(1f, 600f)] public float recoverySeconds = 45f;

    [Range(0.2f, 4f)] public float fillCurve = 1.4f;

    [Range(0f, 0.5f)] public float residualTrail = 0f;

    [Header("④ 痕迹形状")]
    // 痕迹半宽（米）
    public float trailHalfWidth = 0.3f;

    [Range(0.3f, 6f)] public float profilePow = 2f;

    [Range(0f, 0.9f)] public float noiseAmount = 0.35f;

    public Texture2D grainNoiseTex;

    [Range(0.5f, 100f)] public float noiseTiling = 8f;

    [Range(0.2f, 20f)] public float noiseScale = 2.5f;

    [Header("④c 雪堆（痕迹外侧的隆起）")]
    [Range(0f, 1f)] public float bermHeight = 0.3f;

    [Range(0.05f, 3f)] public float bermWidth = 0.5f;

    // 路径两端不生成雪堆的留白（米）
    public float bermEdgeTrim = 0.35f;

    [Range(0f, 0.45f)] public float bermEndTrim = 0.18f;

    [Range(0f, 0.6f)] public float edgeJitter = 0.25f;

    [Header("④c2 深度随机（hash）")]
    [Range(0f, 0.7f)] public float depthVariation = 0.3f;

    [Range(0f, 0.5f)] public float grainAmount = 0.1f;

    [Range(0.5f, 80f)] public float grainScale = 25f;

    [Header("④b 用碰撞体半径（解决“痕迹先于碰撞体出现”）")]
    // 直接用胶囊半径当半宽，痕迹前沿正好和身体对齐
    public bool matchColliderRadius = true;

    public CapsuleCollider sourceCapsule;

    public float widthOffset = 0f;

    [Header("⑤ 材质参数")]
             "saturate(高度×_HeightIntensity − trail×_TrailIntensity)，" +
             "如果这个值比 _HeightIntensity 大太多，结果会被夹到 0，渐变会消失")]
    // 写进材质的痕迹深度
    public float trailDepth = 0.12f;

    [Header("⑥ 方向修正")]
    public bool flipHorizontal = false;
    public bool flipVertical = false;

    [Header("⑦ 采样来源")]
    // 踩踏来源（玩家），留空会自动找
    public Transform trailSource;

    [Header("⑧ 调试")]
    // 屏幕角落的画布预览，调参时用
    public bool showCanvas = true;

    [Range(0.1f, 1.2f)] public float previewSizeRatio = 1f;

    // 预览 Quad 离相机多远
    public float canvasDistance = 1.2f;

    public Vector2 previewOffset = new Vector2(0f, 0f);

    // 预览板放左半边，方便左右对比
    public bool canvasLeftHalf = true;

    public bool verboseLog = false;

    private ComputeShader _compute;
    private int _kernelIndex;

    private RenderTexture _fieldRT;
    private RenderTexture _canvasRT;
    private Material _canvasMat;

    private ComputeBuffer _pointBuffer;
    private Vector4[] _pointData;
    private float[] _pointAges;
    private int _activePointCount;

    private Material[] _snowMaterials;
    private string _trailMapProp = "_TrailMap";
    private string _trailScaleProp = "_TrailScale";
    private string _trailSTProp = "_TrailMap_ST";
    private bool _propsBound;

    private Vector3 _prevPosition;
    private Vector3 _fieldMin;
    private float _fieldSizeCurrent = 1f;
    private bool _initialized;

    void OnEnable() { BuildResources(); }
    void OnDisable() { ReleaseResources(); }

    // 建 RT / buffer / 材质，并把雪地和踩踏来源找齐
    void BuildResources()
    {
        if (_initialized) return;

        _compute = Resources.Load<ComputeShader>("SnowTrailField");
        if (_compute == null)
        {
            var all = Resources.FindObjectsOfTypeAll<ComputeShader>();
            foreach (var c in all)
                if (c != null && c.name == "SnowTrailField") { _compute = c; break; }
        }

        if (_compute == null)
        {
            Debug.LogError("[雪地痕迹] 找不到 ComputeShader 'SnowTrailField'。\n" +
                           "请把 SnowTrailField.compute 放到 Assets 里（建议 Assets/Resources/ 下）。");
            return;
        }

        _kernelIndex = _compute.FindKernel("CSMain");

        if (snowTarget == null) AutoFindSnowTarget();
        if (snowTarget == null)
        {
            Debug.LogError("[雪地痕迹] 没找到雪地物体，请把雪地拖进 Snow Renderer。");
            return;
        }

        _snowMaterials = Application.isPlaying ? snowTarget.materials : snowTarget.sharedMaterials;

        UpdateFieldBounds();

        _fieldRT = new RenderTexture(resolution, resolution, 0, RenderTextureFormat.RFloat)
        {
            name = "SnowTrailFieldRT",
            enableRandomWrite = true,
            wrapMode = TextureWrapMode.Clamp,
            filterMode = FilterMode.Bilinear,
            useMipMap = false,
            autoGenerateMips = false
        };
        _fieldRT.Create();

        _pointData = new Vector4[maxPoints];
        _pointAges = new float[maxPoints];
        _pointBuffer = new ComputeBuffer(maxPoints, 16);

        Shader previewShader = Shader.Find("Hidden/SnowTrailPreview");
        if (previewShader != null)
            _canvasMat = new Material(previewShader) { hideFlags = HideFlags.HideAndDontSave };

        _canvasRT = new RenderTexture(resolution, resolution, 0,
            RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear)
        {
            name = "SnowTrailPreview",
            enableRandomWrite = true,
            wrapMode = TextureWrapMode.Clamp,
            filterMode = FilterMode.Bilinear,
            hideFlags = HideFlags.HideAndDontSave
        };
        _canvasRT.Create();

        if (trailSource == null) AutoFindSource();
        _prevPosition = trailSource != null ? trailSource.position : transform.position;

        if (trailSource == null)
        {
            Debug.LogWarning("[雪地痕迹] 没找到 Stamp Source（玩家），痕迹不会产生。\n" +
                             "请把 Capsule / Character 拖进 Stamp Source 字段。");
        }
        else
        {
            Debug.Log("[雪地痕迹] 踩踏来源 = " + trailSource.name);
        }

        DetectMaterialProps();
        PushMaterialSettings();

        _initialized = true;

        Debug.Log("[雪地痕迹] 路径距离场版初始化完成\n" +
                  "  雪地     : " + snowTarget.name + "\n" +
                  "  RT       : " + resolution + "x" + resolution + " RFloat（compute 写入）\n" +
                  "  区域     : " + _fieldSizeCurrent.ToString("F1") + " 米，中心 " + fieldCenter + "\n" +
                  "  路径     : 最多 " + maxPoints + " 点，间距 " + pointSpacing + " 米\n" +
                  "  恢复     : 每段约 " + recoverySeconds + " 秒填平（曲线 " + fillCurve + "）\n" +
                  "  痕迹半宽 : " + trailHalfWidth.ToString("F2") + " 米　截面指数 " + profilePow);
    }

    // 释放所有运行时资源（RT / buffer / 材质 / 预览物体）
    void ReleaseResources()
    {
        if (_snowMaterials != null)
            foreach (var m in _snowMaterials)
                if (m != null && m.HasProperty(_trailMapProp)) m.SetTexture(_trailMapProp, null);

        if (_pointBuffer != null) { _pointBuffer.ReleaseResources(); _pointBuffer = null; }
        if (_canvasQuad != null) { DestroyImmediate(_canvasQuad); _canvasQuad = null; }
        if (_canvasQuadMat != null) { DestroyImmediate(_canvasQuadMat); _canvasQuadMat = null; }
        if (_fieldRT != null) { _fieldRT.ReleaseResources(); DestroyImmediate(_fieldRT); _fieldRT = null; }
        if (_canvasRT != null) { _canvasRT.ReleaseResources(); DestroyImmediate(_canvasRT); _canvasRT = null; }
        if (_canvasMat != null) { DestroyImmediate(_canvasMat); _canvasMat = null; }

        _snowMaterials = null;
        _propsBound = false;
        _initialized = false;
    }

    // 每帧：记录路径点 → 老化 → 上传 → 重建距离场 → 推给材质
    void Update()
    {
        if (!_initialized) { BuildResources(); return; }
        if (!Application.isPlaying) return;

        UpdateFieldBounds();

        if (trailSource != null)
        {
            Vector3 p = trailSource.position;
            Vector3 d = p - _prevPosition;
            d.y = 0f;

            if (d.magnitude >= Mathf.Max(0.005f, pointSpacing))
            {
                RecordPoint(p);
                _prevPosition = p;
            }
        }

        SyncWidthFromCapsule();

        if (Input.GetKeyDown(KeyCode.P)) ExportCanvasPNG();

        AgePoints();
        PushPoints();

        RebuildField();

        BindTrailMap();

        UpdateCanvas();
    }

    // 记录一个路径点（移动够远才记，避免点太密）
    void RecordPoint(Vector3 world)
    {
        if (_activePointCount >= maxPoints)
        {
            for (int i = 1; i < maxPoints; i++)
            {
                _pointData[i - 1] = _pointData[i];
                _pointAges[i - 1] = _pointAges[i];
            }
            _activePointCount = maxPoints - 1;
        }

        float rnd = Hash01(world.x * 97.3f, world.z * 57.1f);
        float depthScale = 1f + (rnd - 0.5f) * 2f * Mathf.Clamp(depthVariation, 0f, 0.9f);

        _pointData[_activePointCount] = new Vector4(world.x, world.z, 0f, depthScale);
        _pointAges[_activePointCount] = Time.time;
        _activePointCount++;

        if (verboseLog)
            Debug.Log("[雪地痕迹] 记录路径点 " + _activePointCount + "/" + maxPoints +
                      "  世界 " + world.ToString("F2") +
                      "  UV 范围检查: 区域中心 " + fieldCenter.ToString("F1"));
    }

    // 按年龄淘汰路径点：超过痕迹寿命的点从路径里移除
    void AgePoints()
    {
        if (_activePointCount == 0) return;

        float lifetime = Mathf.Max(1f, recoverySeconds);

        for (int i = 0; i < _activePointCount; i++)
        {
            float raw = Mathf.Clamp01((Time.time - _pointAges[i]) / lifetime);

            float aged = Mathf.Pow(raw, Mathf.Max(0.05f, fillCurve));
            aged *= (1f - Mathf.Clamp01(residualTrail));

            _pointData[i].z = aged;
        }

        int drop = 0;
        while (drop < _activePointCount && _pointData[drop].z >= 1f) drop++;

        if (drop > 0)
        {
            for (int i = drop; i < _activePointCount; i++)
            {
                _pointData[i - drop] = _pointData[i];
                _pointAges[i - drop] = _pointAges[i];
            }
            _activePointCount -= drop;
        }
    }

    // 把路径点写进 StructuredBuffer
    void PushPoints()
    {
        if (_pointBuffer == null || _activePointCount <= 0) return;
        _pointBuffer.SetData(_pointData, 0, 0, _activePointCount);
    }

    // dispatch compute，整张距离场重算一遍
    void RebuildField()
    {
        if (_pointBuffer == null) return;

        _compute.SetBuffer(_kernelIndex, "_Points", _pointBuffer);
        _compute.SetInt("_PointCount", _activePointCount);
        _compute.SetVector("_AreaMin", new Vector4(_fieldMin.x, _fieldMin.z, 0f, 0f));
        _compute.SetFloat("_AreaSize", _fieldSizeCurrent);
        _compute.SetInt("_Resolution", resolution);
        _compute.SetFloat("_HalfWidth", Mathf.Max(0.01f, trailHalfWidth));
        _compute.SetFloat("_ProfilePow", Mathf.Max(0.1f, profilePow));
        _compute.SetFloat("_NoiseAmount", Mathf.Clamp01(noiseAmount));
        _compute.SetFloat("_NoiseScale", Mathf.Max(0.01f, noiseScale));
        _compute.SetFloat("_NoiseTiling", Mathf.Max(0.1f, noiseTiling));
        _compute.SetFloat("_BermHeight", Mathf.Clamp01(bermHeight));
        _compute.SetFloat("_BermWidth", Mathf.Max(0.02f, bermWidth));
        float trimFrac = bermEndTrim;
        if (bermEdgeTrim > 0f)
        {
            float pathLen = Mathf.Max(0.01f, (_activePointCount - 1) * Mathf.Max(0.005f, pointSpacing));
            trimFrac = Mathf.Clamp(bermEdgeTrim / pathLen, 0f, 0.45f);
        }
        _compute.SetFloat("_BermEndTrim", trimFrac);
        _compute.SetFloat("_GrainAmount", Mathf.Clamp01(grainAmount));
        _compute.SetFloat("_GrainScale", Mathf.Max(0.1f, grainScale));
        _compute.SetFloat("_EdgeJitter", Mathf.Clamp(edgeJitter, 0f, 0.8f));

        if (grainNoiseTex == null) grainNoiseTex = BuildDefaultNoise();

        if (grainNoiseTex != null)
        {
            _compute.SetTexture(_kernelIndex, "_NoiseTex", grainNoiseTex);
            _compute.SetFloat("_UseNoiseTex", 1f);
        }
        else
        {
            _compute.SetFloat("_UseNoiseTex", 0f);
        }
        _compute.SetTexture(_kernelIndex, "_Result", _fieldRT);

        if (_canvasRT != null)
            _compute.SetTexture(_kernelIndex, "_ResultColor", _canvasRT);

        int groups = Mathf.CeilToInt(resolution / 8f);
        _compute.RebuildField(_kernelIndex, groups, groups, 1);
    }

    // 区域跟随玩家时重新计算覆盖范围
    void UpdateFieldBounds()
    {
        if (useFixedArea)
        {
            _fieldSizeCurrent = Mathf.Max(1f, fieldSize);
        }
        else if (snowTarget != null)
        {
            Bounds b = snowTarget.bounds;
            _fieldSizeCurrent = Mathf.Max(b.size.x, b.size.z);
            fieldSize = _fieldSizeCurrent;

            if (followPlayer)
                fieldCenter = new Vector3(b.center.x, 0f, b.center.z);
        }
        else
        {
            _fieldSizeCurrent = Mathf.Max(1f, fieldSize);
        }

        _fieldMin = new Vector3(fieldCenter.x - _fieldSizeCurrent * 0.5f, 0f,
                               fieldCenter.z - _fieldSizeCurrent * 0.5f);
    }

    // 兼容不同雪 shader 的属性名（_TrailMap / _trail 等）
    void DetectMaterialProps()
    {
        if (_propsBound || _snowMaterials == null) return;

        foreach (var m in _snowMaterials)
        {
            if (m == null) continue;

            if (m.HasProperty("_TrailMap")) _trailMapProp = "_TrailMap";
            else if (m.HasProperty("_trail")) _trailMapProp = "_trail";

            if (m.HasProperty("_TrailScale")) _trailScaleProp = "_TrailScale";
            else if (m.HasProperty("_TrailIntensity")) _trailScaleProp = "_TrailIntensity";

            if (m.HasProperty("_TrailMap_ST")) _trailSTProp = "_TrailMap_ST";
            else if (m.HasProperty("_trail_ST")) _trailSTProp = "_trail_ST";

            _propsBound = true;
            break;
        }

        Debug.Log("[雪地痕迹] 属性名识别：纹理 " + _trailMapProp + " / 强度 " + _trailScaleProp + " / UV " + _trailSTProp);
    }

    // 把痕迹强度和 UV 变换写进材质
    void PushMaterialSettings()
    {
        if (_snowMaterials == null) return;

        foreach (var m in _snowMaterials)
        {
            if (m == null) continue;

            if (m.HasProperty("_HeightIntensity"))
            {
                float hi = m.GetFloat("_HeightIntensity");

                if (hi > 0.0001f && trailDepth > hi * 0.6f)
                {
                    float clamped = hi * 0.35f;
                    Debug.LogWarning("[雪地痕迹] Trail Strength " + trailDepth +
                                     " 太大（_HeightIntensity = " + hi + "），" +
                                     "已被自动钳制为 " + clamped.ToString("F3") + "，否则渐变会被 saturate 夹平。");
                    trailDepth = clamped;
                }

                Debug.Log("[雪地痕迹] 材质参数：_HeightIntensity = " + hi +
                          "，_TrailIntensity = " + trailDepth +
                          (trailDepth > hi ? "  ⚠ 会被夹平" : "  ✅ 在渐变区间内"));
            }

            if (m.HasProperty(_trailScaleProp)) m.SetFloat(_trailScaleProp, trailDepth);
        }
    }

    // 把距离场 RT 绑到雪地材质上
    void BindTrailMap()
    {
        if (_snowMaterials == null || _fieldRT == null) return;

        Vector4 st = new Vector4(flipHorizontal ? -1f : 1f, flipVertical ? -1f : 1f,
                                 flipHorizontal ? 1f : 0f, flipVertical ? 1f : 0f);

        foreach (var m in _snowMaterials)
        {
            if (m == null) continue;
            if (m.HasProperty(_trailMapProp)) m.SetTexture(_trailMapProp, _fieldRT);
            if (m.HasProperty(_trailSTProp)) m.SetVector(_trailSTProp, st);
        }
    }

    // 用胶囊半径同步痕迹半宽
    void SyncWidthFromCapsule()
    {
        if (!matchColliderRadius) return;

        if (sourceCapsule == null)
        {
            if (trailSource != null)
                sourceCapsule = trailSource.GetComponent<CapsuleCollider>();

            if (sourceCapsule == null && trailSource != null)
                sourceCapsule = trailSource.GetComponentInChildren<CapsuleCollider>();

            if (sourceCapsule == null)
                return;
        }

        Vector3 s = sourceCapsule.transform.lossyScale;
        float worldRadius = sourceCapsule.radius * Mathf.Max(Mathf.Abs(s.x), Mathf.Abs(s.z));

        trailHalfWidth = Mathf.Max(0.01f, worldRadius + widthOffset);
    }

    // 没指定踩踏来源就按 Tag 找一个
    void AutoFindSource()
    {
        foreach (var t in Object.FindObjectsOfType<Transform>())
        {
            string n = t.name.ToLowerInvariant();
            if (n.Contains("capsule") || n.Contains("player") || n.Contains("character"))
            {
                trailSource = t;
                Debug.Log("[雪地痕迹] 自动找到踩踏来源：" + t.name);
                return;
            }
        }

        if (Camera.main != null)
        {
            trailSource = Camera.main.transform;
            Debug.Log("[雪地痕迹] 没找到玩家，改用 Main Camera 作为踩踏来源");
        }
    }

    // 没给噪声贴图就程序化生成一张
    static Texture2D BuildDefaultNoise()
    {
        const int size = 256;

        var tex = new Texture2D(size, size, TextureFormat.RGBA32, false)
        {
            name = "SnowTrailDefaultNoise",
            wrapMode = TextureWrapMode.Repeat,
            filterMode = FilterMode.Bilinear,
            hideFlags = HideFlags.HideAndDontSave
        };

        var px = new Color32[size * size];
        var rnd = new System.Random(12345);

        float[,] oct1 = BuildOctave(size, 16, rnd);
        float[,] oct2 = BuildOctave(size, 37, rnd);
        float[,] oct3 = BuildOctave(size, 83, rnd);

        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                float v = oct1[x, y] * 0.55f + oct2[x, y] * 0.3f + oct3[x, y] * 0.15f;
                v = Mathf.Clamp01(v);

                byte b = (byte)(v * 255f);
                px[y * size + x] = new Color32(b, b, b, 255);
            }
        }

        tex.SetPixels32(px);
        tex.Apply(false, true);
        return tex;
    }

    static float[,] BuildOctave(int size, int cells, System.Random rnd)
    {
        var grid = new float[cells, cells];
        for (int y = 0; y < cells; y++)
            for (int x = 0; x < cells; x++)
                grid[x, y] = (float)rnd.NextDouble();

        var result = new float[size, size];
        float step = (float)cells / size;

        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                float fx = x * step;
                float fy = y * step;

                int x0 = (int)fx % cells;
                int y0 = (int)fy % cells;
                int x1 = (x0 + 1) % cells;
                int y1 = (y0 + 1) % cells;

                float tx = fx - Mathf.Floor(fx);
                float ty = fy - Mathf.Floor(fy);
                tx = tx * tx * (3f - 2f * tx);
                ty = ty * ty * (3f - 2f * ty);

                float a = Mathf.Lerp(grid[x0, y0], grid[x1, y0], tx);
                float b = Mathf.Lerp(grid[x0, y1], grid[x1, y1], tx);
                result[x, y] = Mathf.Lerp(a, b, ty);
            }
        }

        return result;
    }

    static float Hash01(float x, float y)
    {
        float h = Mathf.Sin(x * 12.9898f + y * 78.233f) * 43758.5453f;
        return h - Mathf.Floor(h);
    }

    // 没指定雪地就找带雪地材质的渲染器
    void AutoFindSnowTarget()
    {
        if (snowTarget != null) return;

        foreach (var r in Object.FindObjectsOfType<Renderer>())
        {
            if (r == null) continue;
            foreach (var m in r.sharedMaterials)
            {
                if (m == null || m.shader == null) continue;
                string n = m.shader.name;
                if (n == "111" || n.Contains("111") || m.HasProperty("_trail"))
                {
                    snowTarget = r;
                    Debug.Log("[雪地痕迹] 自动找到雪地：" + r.name + "（shader " + n + "）");
                    return;
                }
            }
        }
    }

    private bool _warnedCamera;
    private bool _warnedRT;
    private GameObject _canvasQuad;
    private Material _canvasQuadMat;

    // 屏幕角落的预览：相机前贴一张 Quad，Overlay 队列
    void UpdateCanvas()
    {
        if (!showCanvas)
        {
            if (_canvasQuad != null) _canvasQuad.SetActive(false);
            return;
        }

        Camera cam = Camera.main;
        if (cam == null) cam = Object.FindObjectOfType<Camera>();

        if (cam == null)
        {
            if (!_warnedCamera) { _warnedCamera = true; Debug.LogWarning("[雪地痕迹] 场景里没有可用相机，预览板无法创建。"); }
            return;
        }

        if (_canvasRT == null)
        {
            if (!_warnedRT) { _warnedRT = true; Debug.LogWarning("[雪地痕迹] _previewRT 为空（Init 没跑成功？），预览板无法创建。"); }
            return;
        }

        if (_canvasQuad == null)
        {
            _canvasQuad = GameObject.CreatePrimitive(PrimitiveType.Quad);
            _canvasQuad.name = "SnowTrailPreviewQuad";
            _canvasQuad.hideFlags = HideFlags.DontSave;

            Collider col = _canvasQuad.GetComponent<Collider>();
            if (col != null) Destroy(col);

            _canvasQuad.transform.SetParent(cam.transform, false);

            Shader s = Shader.Find("Hidden/SnowTrailPreviewQuad");
            if (s == null) s = Shader.Find("Universal Render Pipeline/Unlit");

            _canvasQuadMat = new Material(s) { hideFlags = HideFlags.HideAndDontSave };

            var rend = _canvasQuad.GetComponent<Renderer>();
            rend.material = _canvasQuadMat;
            rend.shadowCastingMode = ShadowCastingMode.Off;
            rend.receiveShadows = false;

            Debug.Log("[雪地痕迹] 预览显示板已创建（相机子物体，Overlay 队列）");
        }

        _canvasQuad.SetActive(true);

        float dist = Mathf.Max(0.3f, canvasDistance);
        float halfH = dist * Mathf.Tan(cam.fieldOfView * 0.5f * Mathf.Deg2Rad);
        float halfW = halfH * Mathf.Max(0.1f, cam.aspect);

        float h = Mathf.Clamp(previewSizeRatio, 0.05f, 1.2f) * halfH * 2f;
        float halfBoard = h * 0.5f;
        float margin = 0.02f * halfH;

        Transform t = _canvasQuad.transform;
        float ox = previewOffset.x * halfW;
        float oy = previewOffset.y * halfH;

        if (canvasLeftHalf)
        {
            float availMin = Mathf.Min(halfW, halfH * 2f);
            float side = Mathf.Clamp(previewSizeRatio, 0.05f, 1.2f) * availMin;
            float hb = side * 0.5f;

            float x = -halfW + hb + margin + ox;
            float y = oy;

            t.localPosition = new Vector3(x, y, dist);
            t.localRotation = Quaternion.identity;
            t.localScale = new Vector3(side, side, 1f);
        }
        else
        {
            t.localPosition = new Vector3(
                -(halfW - halfBoard - margin) + ox,
                 (halfH - halfBoard - margin) - oy,
                 dist);
            t.localRotation = Quaternion.identity;
            t.localScale = new Vector3(h, h, 1f);
        }

        if (_canvasQuadMat != null)
        {
            if (_canvasQuadMat.HasProperty("_MainTex")) _canvasQuadMat.SetTexture("_MainTex", _canvasRT);
            if (_canvasQuadMat.HasProperty("_BaseMap")) _canvasQuadMat.SetTexture("_BaseMap", _canvasRT);
        }
    }

    // 把画布导出成 PNG，方便离线看痕迹形状
    public void ExportCanvasPNG()
    {
        if (_canvasRT == null)
        {
            Debug.LogError("[雪地痕迹] 预览图不存在。");
            return;
        }

        RenderTexture prev = RenderTexture.active;
        RenderTexture.active = _canvasRT;

        var tex = new Texture2D(_canvasRT.width, _canvasRT.height, TextureFormat.RGB24, false);
        tex.ReadPixels(new Rect(0, 0, _canvasRT.width, _canvasRT.height), 0, 0);
        tex.Apply();

        RenderTexture.active = prev;

        string path = System.IO.Path.Combine(Application.persistentDataPath,
            "SnowTrailPreview_" + System.DateTime.Now.ToString("HHmmss") + ".png");

        System.IO.File.WriteAllBytes(path, tex.EncodeToPNG());
        Object.Destroy(tex);

        Debug.Log("[雪地痕迹] 预览已导出：" + path);
    }

    // 清空全部痕迹
    public void ClearAll()
    {
        _activePointCount = 0;
        if (_pointBuffer != null) PushPoints();
        if (_fieldRT != null)
        {
            RenderTexture prev = RenderTexture.active;
            RenderTexture.active = _fieldRT;
            GL.Clear(true, true, Color.black);
            RenderTexture.active = prev;
        }
    }

    // 选中时画出距离场范围和路径点
    void OnDrawGizmosSelected()
    {
        Gizmos.color = new Color(0.4f, 0.8f, 1f, 0.9f);
        Gizmos.DrawWireCube(new Vector3(fieldCenter.x, fieldCenter.y, fieldCenter.z),
                            new Vector3(fieldSize, 0.1f, fieldSize));

        Gizmos.color = new Color(1f, 0.6f, 0.2f, 1f);
        for (int i = 1; i < _activePointCount; i++)
        {
            Gizmos.DrawLine(new Vector3(_pointData[i - 1].x, 1f, _pointData[i - 1].y),
                            new Vector3(_pointData[i].x, 1f, _pointData[i].y));
        }
    }
}
