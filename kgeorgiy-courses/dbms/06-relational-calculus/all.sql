

--1.1s
select
    StudentId,
    StudentName,
    GroupId
from
    Students
where
    StudentName = :StudentName;


--1.2s
select
    StudentId,
    StudentName,
    GroupId
from
    Students S
where
    GroupId in (
        select
            GroupId
        from
            Groups
        where
            GroupName = :GroupName
    );


--1.3s
select
    Students.StudentId,
    StudentName,
    GroupId
from
    Students,
    Marks
where
    Students.StudentId = Marks.StudentId
    and CourseId = :CourseId
    and Mark = :Mark


--1.4s
select
    S.StudentId,
    StudentName,
    GroupId
from
    Students S,
    Marks M,
    Courses C
where
    S.StudentId = M.StudentId
    and M.CourseId = C.CourseId
    and CourseName = :CourseName
    and Mark = :Mark


--2.1s
select
    StudentId,
    StudentName,
    GroupName
from
    Students S,
    Groups G
where
    S.GroupId = G.GroupId;


--2.2s
select
    StudentId,
    StudentName,
    GroupName
from
    Students S,
    Groups G
where
    S.GroupId = G.GroupId
    and not exists (
        select
            *
        from
            Marks M
        where
            S.StudentId = M.StudentId
            and CourseId = :CourseId
    )


--2.3s
select
    StudentId,
    StudentName,
    GroupName
from
    Students S,
    Groups G
where
    S.GroupId = G.GroupId
    and not exists (
        select
            *
        from
            Marks M,
            Courses C
        where
            S.StudentId = M.StudentId
            and M.CourseId = C.CourseId
            and CourseName = :CourseName
    )


--2.4s
select
    distinct StudentId,
    StudentName,
    GroupName
from
    Students S,
    Groups G,
    Plan P
where
    S.GroupId = G.GroupId
    and S.GroupId = P.GroupId
    and CourseId = :CourseId
    and not exists (
        select
            *
        from
            Marks M
        where
            S.StudentId = M.StudentId
            and CourseId = :CourseId
    )


--2.5s
select
    distinct StudentId,
    StudentName,
    GroupName
from
    Students S,
    Groups G,
    Plan P,
    Courses C
where
    S.GroupId = G.GroupId
    and S.GroupId = P.GroupId
    and P.CourseId = C.CourseId
    and CourseName = :CourseName
    and StudentId not in (
        select
            StudentId
        from
            Marks M,
            Courses C
        where
            M.CourseId = C.CourseId
            and CourseName = :CourseName
    )


--3.1s
select
    distinct StudentId,
    CourseId
from
    Marks
union
select
    distinct StudentId,
    CourseId
from
    Students S,
    Plan P
where
    S.GroupId = P.GroupId;


--3.2s
select
    StudentName,
    CourseName
from
    Students S,
    Courses C
where
    exists (
        select
            *
        from
            Plan P
        where
            P.GroupId = S.GroupId
            and P.CourseId = C.CourseId
        union
        select
            *
        from
            Marks M
        where
            M.StudentId = S.StudentId
            and M.CourseId = C.CourseId
    )


--4.1s
select
    StudentName,
    CourseName
from
    Students S,
    Courses C,
    (
        select
            distinct StudentId,
            CourseId
        from
            Students S,
            Plan P
        where
            S.GroupId = P.GroupId
            and not exists (
                select
                    *
                from
                    Marks M
                where
                    M.StudentId = S.StudentId
                    and M.CourseId = P.CourseId
            )
    ) Q
where
    S.StudentId = Q.StudentId
    and C.CourseId = Q.CourseId


--4.2s
select
    StudentName,
    CourseName
from
    Students S,
    Courses C,
    (
        select
            distinct StudentId,
            CourseId
        from
            Students S,
            Plan P
        where
            S.GroupId = P.GroupId
            and exists (
                select
                    *
                from
                    Marks M
                where
                    M.StudentId = S.StudentId
                    and M.CourseId = P.CourseId
                    and Mark <= 2
            )
    ) Q
where
    S.StudentId = Q.StudentId
    and C.CourseId = Q.CourseId


--4.3s
select
    StudentName,
    CourseName
from
    Students S,
    Courses C,
    (
        select
            distinct StudentId,
            CourseId
        from
            Students S,
            Plan P
        where
            S.GroupId = P.GroupId
            and not exists (
                select
                    *
                from
                    Marks M
                where
                    M.StudentId = S.StudentId
                    and M.CourseId = P.CourseId
                    and Mark > 2
            )
    ) Q
where
    S.StudentId = Q.StudentId
    and C.CourseId = Q.CourseId


--5.1s
select
    distinct S.StudentId
from
    Students S,
    Plan P,
    Lecturers L,
    Marks M
where
    S.GroupId = P.GroupId
    and P.LecturerId = L.LecturerId
    and S.StudentId = M.StudentId
    and P.CourseId = M.CourseId
    and LecturerName = :LecturerName;


--5.2s
select
    StudentId
from
    Students
except
select
    distinct S.StudentId
from
    Students S,
    Plan P,
    Lecturers L,
    Marks M
where
    S.GroupId = P.GroupId
    and P.LecturerId = L.LecturerId
    and S.StudentId = M.StudentId
    and P.CourseId = M.CourseId
    and LecturerName = :LecturerName;


--5.3s
select
    StudentId
from
    Students S
where
    not exists (
        select
            CourseId
        from
            Plan P,
            Lecturers L
        where
            P.LecturerId = L.LecturerId
            and LecturerName = :LecturerName
            and not exists (
                select
                    *
                from
                    Marks M
                where
                    M.StudentId = S.StudentId
                    and M.CourseId = P.CourseId
            )
    )


--5.4s
select
    StudentId
from
    Students S
where
    not exists (
        select
            CourseId
        from
            Plan P,
            Lecturers L
        where
            S.GroupId = P.GroupId
            and P.LecturerId = L.LecturerId
            and LecturerName = :LecturerName
            and not exists (
                select
                    *
                from
                    Marks M
                where
                    M.StudentId = S.StudentId
                    and M.CourseId = P.CourseId
            )
    )


--6.1s
select
    GroupId,
    CourseId
from
    Groups G,
    Courses C
where
    not exists (
        select
            *
        from
            Students S
        where
            S.GroupId = G.GroupId
            and not exists (
                select
                    *
                from
                    Marks M
                where
                    M.StudentId = S.StudentId
                    and M.CourseId = C.CourseId
            )
    )


--6.2s
select
    GroupName,
    CourseName
from
    Groups G,
    Courses C
where
    not exists (
        select
            *
        from
            Students S
        where
            S.GroupId = G.GroupId
            and not exists (
                select
                    *
                from
                    Marks M
                where
                    M.StudentId = S.StudentId
                    and M.CourseId = C.CourseId
            )
    )


--do
